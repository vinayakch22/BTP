"""
=================================================================
LOW-LEVEL GNN EXPLAINABILITY FOR H2GnnDTI
=================================================================

This script computes feature attributions for the low-level GAT
encoder (GNNNet) using Captum Integrated Gradients.

Target: GNNNet (NodeRepresentation.py)
  - Drug branch: 3 GAT layers → global_mean_pool → FC → 160-dim
  - Protein branch: 3 GAT layers → global_mean_pool → FC → 160-dim

What we explain:
  - Which atoms are important in the drug molecular graph
  - Which bonds are important in the drug graph
  - Which residues are important in the protein contact graph
  - Which contacts are important in the protein graph

Method: Captum Integrated Gradients (primary)
=================================================================
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from captum.attr import IntegratedGradients

from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool as gep

# Local imports
from NodeRepresentation import GNNNet
from model import H2GNN
from data_load import dataload
from setting import process
import opt
import scipy.sparse as sp


# =====================================================================
# SECTION 1: WRAPPER MODELS FOR CAPTUM
# =====================================================================

class DrugGATWrapper(nn.Module):
    """
    Wraps the drug branch of GNNNet for Captum attribution.
    
    Computes: drug_features → GAT layers → pool → FC → embedding → L2 norm
    
    The L2 norm of the embedding is used as the scalar target because:
    1. It measures the overall "activation strength" of the drug representation
    2. Gradients flow cleanly through the GAT → pool → FC path
    3. This directly explains which atoms/features contribute to the 
       drug's learned representation in the heterogeneous graph
    """
    def __init__(self, gnnnet, edge_index):
        super().__init__()
        self.gnnnet = gnnnet
        self.edge_index = edge_index

    def forward(self, drug_features):
        num_atoms = drug_features.shape[0]
        batch_vec = torch.zeros(num_atoms, dtype=torch.long, device=drug_features.device)

        x = F.relu(self.gnnnet.mol_conv1(drug_features, self.edge_index))
        x = F.relu(self.gnnnet.mol_conv2(x, self.edge_index))
        x = F.relu(self.gnnnet.mol_conv3(x, self.edge_index))
        x = gep(x, batch_vec)  # (1, 312)
        x = F.relu(self.gnnnet.mol_fc_g1(x))
        x = self.gnnnet.mol_fc_g2(x)
        # Return L2 norm of embedding as scalar target
        return torch.norm(x, dim=1)


class ProteinGATWrapper(nn.Module):
    """
    Wraps the protein branch of GNNNet for Captum attribution.
    
    Computes: pro_features → GAT layers → pool → FC → embedding → L2 norm
    """
    def __init__(self, gnnnet, edge_index):
        super().__init__()
        self.gnnnet = gnnnet
        self.edge_index = edge_index

    def forward(self, pro_features):
        num_residues = pro_features.shape[0]
        batch_vec = torch.zeros(num_residues, dtype=torch.long, device=pro_features.device)

        xt = F.relu(self.gnnnet.pro_conv1(pro_features, self.edge_index))
        xt = F.relu(self.gnnnet.pro_conv2(xt, self.edge_index))
        xt = F.relu(self.gnnnet.pro_conv3(xt, self.edge_index))
        xt = gep(xt, batch_vec)  # (1, 216)
        xt = F.relu(self.gnnnet.pro_fc_g1(xt))
        xt = self.gnnnet.pro_fc_g2(xt)
        # Return L2 norm of embedding as scalar target
        return torch.norm(xt, dim=1)


# =====================================================================
# SECTION 2: ATTRIBUTION COMPUTATION
# =====================================================================

def compute_drug_attributions(wrapper, drug_features, n_steps=200):
    """
    Compute Integrated Gradients attributions for drug atom features.

    Args:
        wrapper: EndToEndDrugWrapper model
        drug_features: (num_atoms, 78) tensor
        n_steps: number of IG interpolation steps

    Returns:
        attributions: (num_atoms, 78) attribution matrix
        node_importance: (num_atoms,) aggregated per-node importance
    """
    ig = IntegratedGradients(wrapper)

    # Baseline: zero features (absence of all atom properties)
    baseline = torch.zeros_like(drug_features)

    attributions = ig.attribute(
        drug_features,
        baselines=baseline,
        n_steps=n_steps,
        method='gausslegendre',
        return_convergence_delta=False
    )

    # Aggregate: node importance = sum of absolute feature attributions
    node_importance = torch.sum(torch.abs(attributions), dim=1)

    return attributions, node_importance


def compute_protein_attributions(wrapper, pro_features, n_steps=200):
    """
    Compute Integrated Gradients attributions for protein residue features.

    Args:
        wrapper: EndToEndProteinWrapper model
        pro_features: (num_residues, 54) tensor
        n_steps: number of IG interpolation steps

    Returns:
        attributions: (num_residues, 54) attribution matrix
        node_importance: (num_residues,) aggregated per-node importance
    """
    ig = IntegratedGradients(wrapper)

    baseline = torch.zeros_like(pro_features)

    attributions = ig.attribute(
        pro_features,
        baselines=baseline,
        n_steps=n_steps,
        method='gausslegendre',
        return_convergence_delta=False
    )

    node_importance = torch.sum(torch.abs(attributions), dim=1)

    return attributions, node_importance


def compute_edge_importance(node_importance, edge_index):
    """
    Approximate edge importance from node importance.
    importance(edge i->j) = node_importance[i] + node_importance[j]

    Args:
        node_importance: (num_nodes,) tensor
        edge_index: (2, num_edges) tensor

    Returns:
        edge_importance: (num_edges,) tensor
        top_edges: list of (src, dst, importance) sorted descending
    """
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]

    edge_imp = node_importance[src_nodes] + node_importance[dst_nodes]

    # Sort and get top edges (remove self-loops and duplicate directions)
    seen = set()
    top_edges = []
    sorted_indices = torch.argsort(edge_imp, descending=True)

    for idx in sorted_indices:
        s = src_nodes[idx].item()
        d = dst_nodes[idx].item()
        if s == d:
            continue
        edge_key = (min(s, d), max(s, d))
        if edge_key not in seen:
            seen.add(edge_key)
            top_edges.append((s, d, edge_imp[idx].item()))

    return edge_imp, top_edges


# =====================================================================
# SECTION 3: VISUALIZATION
# =====================================================================

def visualize_drug_graph(smiles, node_importance, edge_importance_list,
                         drug_id, save_dir, sample_id):
    """
    Visualize drug molecule with atoms colored by importance.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  [WARNING] Could not parse SMILES: {smiles[:50]}...")
        return

    # Normalize importance to [0, 1]
    imp = node_importance.cpu().numpy()
    if imp.max() > imp.min():
        imp_norm = (imp - imp.min()) / (imp.max() - imp.min())
    else:
        imp_norm = np.zeros_like(imp)

    # Truncate if molecule has fewer atoms than importance vector
    num_atoms = mol.GetNumAtoms()
    imp_norm = imp_norm[:num_atoms]

    # Create atom color map (red = important, white = unimportant)
    cmap = cm.get_cmap('Reds')
    atom_colors = {}
    atom_radii = {}
    for i in range(num_atoms):
        rgba = cmap(imp_norm[i])
        atom_colors[i] = rgba
        atom_radii[i] = 0.3 + 0.4 * imp_norm[i]

    # Highlight top edges (bonds)
    highlight_bonds = []
    bond_colors = {}
    if edge_importance_list:
        top_bond_edges = edge_importance_list[:10]
        for src, dst, _ in top_bond_edges:
            if src < num_atoms and dst < num_atoms:
                bond_idx = mol.GetBondBetweenAtoms(src, dst)
                if bond_idx is not None:
                    highlight_bonds.append(bond_idx.GetIdx())
                    bond_colors[bond_idx.GetIdx()] = (0.8, 0.2, 0.2, 0.8)

    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)

    # Draw molecule
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Molecule with atom importance coloring
    drawer = Draw.MolDraw2DSVG(600, 400)
    drawer.drawOptions().addAtomIndices = True
    highlight_atoms = list(range(num_atoms))
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors
    )
    drawer.FinishDrawing()

    svg_text = drawer.GetDrawingText()
    svg_path = os.path.join(save_dir, f'sample_{sample_id}_drug_{drug_id}_mol.svg')
    with open(svg_path, 'w') as f:
        f.write(svg_text)

    # Right: Bar chart of atom importance
    axes[0].bar(range(num_atoms), imp[:num_atoms], color=[cmap(v) for v in imp_norm])
    axes[0].set_xlabel('Atom Index')
    axes[0].set_ylabel('Importance Score')
    axes[0].set_title(f'Drug {drug_id}: Atom Importance')

    # Add atom symbols as labels
    atom_labels = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(num_atoms)]
    axes[0].set_xticks(range(num_atoms))
    if num_atoms <= 40:
        axes[0].set_xticklabels([f'{i}:{s}' for i, s in enumerate(atom_labels)],
                                 rotation=90, fontsize=6)

    # Right panel: Top atoms table
    axes[1].axis('off')
    sorted_idx = np.argsort(imp[:num_atoms])[::-1]
    table_data = []
    for rank, idx in enumerate(sorted_idx[:15]):
        table_data.append([
            f'{rank+1}',
            f'Atom {idx}',
            atom_labels[idx],
            f'{imp[idx]:.4f}'
        ])
    if table_data:
        table = axes[1].table(
            cellText=table_data,
            colLabels=['Rank', 'Atom', 'Element', 'Importance'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
    axes[1].set_title(f'Top 15 Important Atoms')

    plt.suptitle(f'Sample {sample_id} — Drug {drug_id} Atom Importance', fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, f'sample_{sample_id}_drug_{drug_id}_importance.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved drug visualization: {fig_path}")
    print(f"  Saved drug molecule SVG:  {svg_path}")


def visualize_protein_importance(residue_importance, protein_id, protein_seq,
                                 edge_index, save_dir, sample_id):
    """
    Visualize protein residue importance as bar chart + contact map heatmap.
    """
    imp = residue_importance.cpu().numpy()
    num_residues = len(imp)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: Residue importance bar chart
    if imp.max() > imp.min():
        imp_norm = (imp - imp.min()) / (imp.max() - imp.min())
    else:
        imp_norm = np.zeros_like(imp)

    cmap = cm.get_cmap('YlOrRd')
    colors = [cmap(v) for v in imp_norm]
    axes[0].bar(range(num_residues), imp, color=colors, width=1.0)
    axes[0].set_xlabel('Residue Index')
    axes[0].set_ylabel('Importance Score')
    axes[0].set_title(f'Protein {protein_id}: Residue Importance ({num_residues} residues)')

    # Mark top 10 residues
    top_indices = np.argsort(imp)[::-1][:10]
    for idx in top_indices:
        if idx < len(protein_seq):
            axes[0].annotate(
                f'{idx}:{protein_seq[idx]}',
                xy=(idx, imp[idx]),
                fontsize=6,
                ha='center',
                va='bottom',
                rotation=45
            )

    # Right: Contact map with importance overlay
    edge_idx_np = edge_index.cpu().numpy()
    contact_matrix = np.zeros((num_residues, num_residues))
    for i in range(edge_idx_np.shape[1]):
        src, dst = edge_idx_np[0, i], edge_idx_np[1, i]
        if src < num_residues and dst < num_residues:
            # Weight by node importance
            contact_matrix[src, dst] = imp[src] + imp[dst]

    im = axes[1].imshow(contact_matrix, cmap='hot', interpolation='nearest')
    axes[1].set_xlabel('Residue Index')
    axes[1].set_ylabel('Residue Index')
    axes[1].set_title(f'Contact Map Weighted by Importance')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle(f'Sample {sample_id} — Protein {protein_id} Residue Importance', fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, f'sample_{sample_id}_protein_{protein_id}_importance.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved protein visualization: {fig_path}")


# =====================================================================
# SECTION 4: DATA RECONSTRUCTION HELPERS
# =====================================================================

def get_drug_graph_data(drug_id, data_new, drugmap):
    """
    Get the individual drug graph (features, edge_index) for a specific drug.
    Reconstructs from SMILES using the same pipeline as setting.py.
    """
    from setting import smile_to_graph

    # Find SMILES for this drug_id
    smiles = None
    for item in data_new:
        if item[0] == drug_id:
            smiles = item[3]
            break

    if smiles is None:
        raise ValueError(f"Drug {drug_id} not found in data")

    c_size, features, edge_index = smile_to_graph(smiles)

    x = torch.FloatTensor(features)
    ei = torch.LongTensor(edge_index).t().contiguous()
    batch = torch.zeros(x.shape[0], dtype=torch.long)

    return Data(x=x, edge_index=ei, batch=batch), smiles


def get_protein_graph_data(protein_id, data_new, dataset):
    """
    Get the individual protein graph (features, edge_index) for a specific protein.
    Reconstructs from contact map using the same pipeline as pro_graph.py.
    """
    from pro_graph import target_to_graph

    # Find sequence for this protein_id
    sequence = None
    for item in data_new:
        if item[1] == protein_id:
            sequence = item[4]
            break

    if sequence is None:
        raise ValueError(f"Protein {protein_id} not found in data")

    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/pconsc4'

    target_size, target_features, target_edge_index = target_to_graph(
        protein_id, sequence, contac_path, msa_path
    )

    x = torch.FloatTensor(target_features)
    ei = torch.LongTensor(target_edge_index).t().contiguous()
    batch = torch.zeros(x.shape[0], dtype=torch.long)

    return Data(x=x, edge_index=ei, batch=batch), sequence


def build_drug_protein_maps(data_new):
    """
    Build the same drug/protein ID → index mappings as setting.py process().
    """
    drugid = list(set([item[0] for item in data_new]))
    drugid.sort()
    proteinid = list(set([item[1] for item in data_new]))
    proteinid.sort()

    drugmap = {did: idx for idx, did in enumerate(drugid)}
    proteinmap = {pid: idx for idx, pid in enumerate(proteinid)}

    return drugmap, proteinmap, drugid, proteinid


def get_interaction_pairs(data_new, drugmap, proteinmap):
    """
    Get all interaction pairs with their labels.
    Returns list of (drug_idx, protein_idx, drug_id, protein_id, label)
    """
    pairs = []
    seen = set()
    for item in data_new:
        drug_id, protein_id, label = item[0], item[1], item[2]
        if drug_id in drugmap and protein_id in proteinmap:
            key = (drug_id, protein_id)
            if key not in seen:
                seen.add(key)
                pairs.append((
                    drugmap[drug_id],
                    proteinmap[protein_id],
                    drug_id,
                    protein_id,
                    label
                ))
    return pairs


def select_test_pairs(all_pairs, idx_test, nb_drugs, nb_proteins, num_samples=100, seed=42):
    """Sample explainability pairs from the held-out test mask only."""
    if torch.is_tensor(idx_test):
        test_mask = idx_test.detach().cpu().numpy().astype(bool)
    else:
        test_mask = np.asarray(idx_test).astype(bool)
    test_mask = test_mask.reshape(nb_drugs, nb_proteins)

    test_pairs = [p for p in all_pairs if test_mask[int(p[0]), int(p[1])]]
    positive_pairs = [p for p in test_pairs if p[4] == 1]
    negative_pairs = [p for p in test_pairs if p[4] == 0]
    print(f"  Test pairs available: {len(positive_pairs)} positive, {len(negative_pairs)} negative")

    if len(test_pairs) == 0:
        raise ValueError("No test pairs available for explainability. Retrain with the leakage-fixed pipeline.")

    np.random.seed(seed)
    n_pos_target = num_samples // 2
    n_neg_target = num_samples - n_pos_target
    if len(positive_pairs) >= n_pos_target and len(negative_pairs) >= n_neg_target:
        n_pos, n_neg = n_pos_target, n_neg_target
    else:
        n_pos = min(n_pos_target, len(positive_pairs))
        n_neg = min(num_samples - n_pos, len(negative_pairs))
        leftover = num_samples - n_pos - n_neg
        extra_pos = min(leftover, max(0, len(positive_pairs) - n_pos))
        n_pos += extra_pos
        leftover -= extra_pos
        n_neg += min(leftover, max(0, len(negative_pairs) - n_neg))

    selected = []
    if n_pos > 0:
        selected += [positive_pairs[i] for i in np.random.choice(len(positive_pairs), n_pos, replace=False)]
    if n_neg > 0:
        selected += [negative_pairs[i] for i in np.random.choice(len(negative_pairs), n_neg, replace=False)]
    np.random.shuffle(selected)
    print(f"  Selected {len(selected)} TEST samples ({n_pos} positive, {n_neg} negative)")
    return selected


# =====================================================================
# SECTION 5: SINGLE SAMPLE EXPLAINABILITY
# =====================================================================

def run_single_sample(sample_idx, drug_idx, protein_idx, drug_id, protein_id,
                      ground_truth, gnnnet, h2gnn, full_features, full_adj,
                      data_new, dataset, nb_drugs, nb_all, device,
                      save_dir, n_steps=200):
    """
    Run the complete explainability pipeline for one drug-protein pair.
    """
    print("=" * 60)
    print("LOW-LEVEL GNN EXPLAINABILITY")
    print("=" * 60)
    print()
    print(f"  Sample ID:    {sample_idx}")
    print(f"  Drug ID:      {drug_id}")
    print(f"  Drug Index:   {drug_idx}")
    print(f"  Protein ID:   {protein_id}")
    print(f"  Protein Index:{protein_idx}")
    print(f"  Ground Truth: {ground_truth}")
    print()

    # --- Get prediction score ---
    gnnnet.eval()
    h2gnn.eval()

    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde, alpha = h2gnn(
            full_features.to(device), full_adj.to(device)
        )
        pred_score = adj_hat[drug_idx, nb_drugs + protein_idx].item()
        pred_score_sigmoid = torch.sigmoid(torch.tensor(pred_score)).item()

    print(f"  Prediction (raw):     {pred_score:.4f}")
    print(f"  Prediction (sigmoid): {pred_score_sigmoid:.4f}")
    print()

    # --- Get individual drug and protein graphs ---
    print("  Loading drug graph...")
    drugmap, proteinmap, _, _ = build_drug_protein_maps(data_new)
    drug_data, smiles = get_drug_graph_data(drug_id, data_new, drugmap)
    drug_data = drug_data.to(device)
    print(f"    SMILES: {smiles[:80]}{'...' if len(smiles)>80 else ''}")
    print(f"    Atoms: {drug_data.x.shape[0]}, Features: {drug_data.x.shape[1]}")
    print(f"    Edges: {drug_data.edge_index.shape[1]}")

    print("  Loading protein graph...")
    pro_data, sequence = get_protein_graph_data(protein_id, data_new, dataset)
    pro_data = pro_data.to(device)
    print(f"    Sequence length: {len(sequence)}")
    print(f"    Residues (nodes): {pro_data.x.shape[0]}, Features: {pro_data.x.shape[1]}")
    print(f"    Contacts (edges): {pro_data.edge_index.shape[1]}")
    print()

    # ---- DRUG ATTRIBUTION ----
    print("  Running Integrated Gradients on Drug Graph...")
    print(f"    Steps: {n_steps}")

    drug_wrapper = DrugGATWrapper(
        gnnnet=gnnnet,
        edge_index=drug_data.edge_index,
    ).to(device)

    drug_features_input = drug_data.x.clone().requires_grad_(True)

    drug_attrs, drug_node_imp = compute_drug_attributions(
        drug_wrapper, drug_features_input, n_steps=n_steps
    )

    # Validate attributions
    assert drug_attrs.shape == drug_data.x.shape, \
        f"Drug attribution shape mismatch: {drug_attrs.shape} vs {drug_data.x.shape}"
    assert torch.any(drug_attrs != 0), "Drug attributions are all zero — gradient flow issue!"

    print(f"    Attribution shape: {drug_attrs.shape}")
    print(f"    Non-zero attributions: {torch.count_nonzero(drug_attrs).item()}")
    print()

    # Print top drug atoms
    drug_node_imp_np = drug_node_imp.detach().cpu().numpy()
    mol = Chem.MolFromSmiles(smiles)
    num_mol_atoms = mol.GetNumAtoms() if mol else drug_data.x.shape[0]

    print("  Drug Graph Analysis")
    print("  " + "-" * 40)
    print("  Top Atoms:")
    sorted_drug_idx = np.argsort(drug_node_imp_np)[::-1]
    for rank, idx in enumerate(sorted_drug_idx[:15]):
        if mol and idx < num_mol_atoms:
            atom_sym = mol.GetAtomWithIdx(int(idx)).GetSymbol()
        else:
            atom_sym = '?'
        print(f"    {rank+1:2d}. Atom {idx:3d} ({atom_sym:>2s}) -> {drug_node_imp_np[idx]:.6f}")
    print()

    # Drug edge importance
    drug_edge_imp, drug_top_edges = compute_edge_importance(
        drug_node_imp.detach().cpu(), drug_data.edge_index.cpu()
    )
    print("  Top Bonds:")
    for rank, (s, d, imp) in enumerate(drug_top_edges[:10]):
        if mol and s < num_mol_atoms and d < num_mol_atoms:
            s_sym = mol.GetAtomWithIdx(s).GetSymbol()
            d_sym = mol.GetAtomWithIdx(d).GetSymbol()
        else:
            s_sym, d_sym = '?', '?'
        print(f"    {rank+1:2d}. Bond {s}({s_sym}) -- {d}({d_sym}) -> {imp:.6f}")
    print()

    # ---- PROTEIN ATTRIBUTION ----
    print("  Running Integrated Gradients on Protein Graph...")
    print(f"    Steps: {n_steps}")

    pro_wrapper = ProteinGATWrapper(
        gnnnet=gnnnet,
        edge_index=pro_data.edge_index,
    ).to(device)

    pro_features_input = pro_data.x.clone().requires_grad_(True)

    pro_attrs, pro_node_imp = compute_protein_attributions(
        pro_wrapper, pro_features_input, n_steps=n_steps
    )

    assert pro_attrs.shape == pro_data.x.shape, \
        f"Protein attribution shape mismatch: {pro_attrs.shape} vs {pro_data.x.shape}"
    assert torch.any(pro_attrs != 0), "Protein attributions are all zero — gradient flow issue!"

    print(f"    Attribution shape: {pro_attrs.shape}")
    print(f"    Non-zero attributions: {torch.count_nonzero(pro_attrs).item()}")
    print()

    # Print top protein residues
    pro_node_imp_np = pro_node_imp.detach().cpu().numpy()
    print("  Protein Graph Analysis")
    print("  " + "-" * 40)
    print("  Top Residues:")
    sorted_pro_idx = np.argsort(pro_node_imp_np)[::-1]
    for rank, idx in enumerate(sorted_pro_idx[:15]):
        if idx < len(sequence):
            res_char = sequence[idx]
        else:
            res_char = '?'
        print(f"    {rank+1:2d}. Residue {idx:4d} ({res_char}) -> {pro_node_imp_np[idx]:.6f}")
    print()

    # Protein edge importance
    pro_edge_imp, pro_top_edges = compute_edge_importance(
        pro_node_imp.detach().cpu(), pro_data.edge_index.cpu()
    )
    print("  Top Contacts:")
    for rank, (s, d, imp) in enumerate(pro_top_edges[:10]):
        if s < len(sequence) and d < len(sequence):
            s_res = sequence[s]
            d_res = sequence[d]
        else:
            s_res, d_res = '?', '?'
        print(f"    {rank+1:2d}. Contact {s}({s_res}) -- {d}({d_res}) -> {imp:.6f}")
    print()

    # ---- VISUALIZATIONS ----
    print("  Saving visualizations...")
    visualize_drug_graph(
        smiles, drug_node_imp.detach().cpu(), drug_top_edges,
        drug_id, save_dir, sample_idx
    )
    visualize_protein_importance(
        pro_node_imp.detach().cpu(), protein_id, sequence,
        pro_data.edge_index, save_dir, sample_idx
    )

    print()
    print("  Completed.")
    print("=" * 60)
    print()

    return {
        'sample_idx': sample_idx,
        'drug_id': drug_id,
        'protein_id': protein_id,
        'ground_truth': ground_truth,
        'prediction': pred_score_sigmoid,
        'drug_node_importance': drug_node_imp_np,
        'protein_node_importance': pro_node_imp_np,
        'drug_top_atoms': sorted_drug_idx[:15].tolist(),
        'protein_top_residues': sorted_pro_idx[:15].tolist(),
    }


# =====================================================================
# SECTION 6: GLOBAL ANALYSIS
# =====================================================================

def run_global_analysis(num_samples, all_results, save_dir):
    """
    Aggregate explainability results across multiple samples.
    """
    print()
    print("=" * 60)
    print("GLOBAL ANALYSIS")
    print("=" * 60)
    print(f"  Analyzed {len(all_results)} samples")
    print()

    # Collect all drug node importance distributions
    all_drug_imp = []
    all_pro_imp = []
    drug_atom_frequency = defaultdict(int)
    pro_residue_frequency = defaultdict(int)

    for result in all_results:
        all_drug_imp.extend(result['drug_node_importance'].tolist())
        all_pro_imp.extend(result['protein_node_importance'].tolist())

        # Count how often each atom/residue appears in top-15
        for atom_idx in result['drug_top_atoms']:
            drug_atom_frequency[atom_idx] += 1
        for res_idx in result['protein_top_residues']:
            pro_residue_frequency[res_idx] += 1

    # --- Plot 1: Attribution Distribution Histograms ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(all_drug_imp, bins=50, color='steelblue', alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Attribution Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Drug Atom Importance Distribution (n={len(all_drug_imp)})')
    axes[0].axvline(np.mean(all_drug_imp), color='red', linestyle='--',
                     label=f'Mean: {np.mean(all_drug_imp):.4f}')
    axes[0].legend()

    axes[1].hist(all_pro_imp, bins=50, color='forestgreen', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Attribution Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Protein Residue Importance Distribution (n={len(all_pro_imp)})')
    axes[1].axvline(np.mean(all_pro_imp), color='red', linestyle='--',
                     label=f'Mean: {np.mean(all_pro_imp):.4f}')
    axes[1].legend()

    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'global_attribution_distributions.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # --- Plot 2: Prediction vs Ground Truth ---
    predictions = [r['prediction'] for r in all_results]
    ground_truths = [r['ground_truth'] for r in all_results]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['green' if gt == 1 else 'red' for gt in ground_truths]
    ax.scatter(range(len(predictions)), predictions, c=colors, alpha=0.6, s=30)
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Prediction Score (sigmoid)')
    ax.set_title('Predictions vs Ground Truth')
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Positive (GT=1)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Negative (GT=0)')
    ])
    fig_path = os.path.join(save_dir, 'global_predictions.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # --- Summary Statistics ---
    print()
    print("  Summary Statistics")
    print("  " + "-" * 40)
    print(f"  Drug atom importance:    mean={np.mean(all_drug_imp):.6f}, "
          f"std={np.std(all_drug_imp):.6f}, max={np.max(all_drug_imp):.6f}")
    print(f"  Protein residue importance: mean={np.mean(all_pro_imp):.6f}, "
          f"std={np.std(all_pro_imp):.6f}, max={np.max(all_pro_imp):.6f}")
    print()

    # --- Most frequently important atoms/residues ---
    print("  Most Frequently Important Drug Atom Indices (across all samples):")
    sorted_atoms = sorted(drug_atom_frequency.items(), key=lambda x: x[1], reverse=True)
    for idx, (atom_idx, count) in enumerate(sorted_atoms[:10]):
        print(f"    {idx+1}. Atom {atom_idx} appeared in top-15: {count}/{len(all_results)} samples")

    print()
    print("  Most Frequently Important Protein Residue Indices (across all samples):")
    sorted_residues = sorted(pro_residue_frequency.items(), key=lambda x: x[1], reverse=True)
    for idx, (res_idx, count) in enumerate(sorted_residues[:10]):
        print(f"    {idx+1}. Residue {res_idx} appeared in top-15: {count}/{len(all_results)} samples")

    print()
    print("=" * 60)
    print("GLOBAL ANALYSIS COMPLETE")
    print("=" * 60)


# =====================================================================
# SECTION 7: MAIN EXECUTION
# =====================================================================

def main():
    print()
    print("=" * 60)
    print("H2GnnDTI LOW-LEVEL GNN EXPLAINABILITY PIPELINE")
    print("=" * 60)
    print()

    # --- Configuration ---
    DATASET = "kiba"
    NUM_SAMPLES = 100
    N_STEPS = 200
    SAVE_DIR = os.path.join('results', 'explainability_lowlevel')
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device('cpu')  # Use CPU for explainability (IG needs gradients)
    print(f"  Dataset:     {DATASET}")
    print(f"  Samples:     {NUM_SAMPLES}")
    print(f"  IG Steps:    {N_STEPS}")
    print(f"  Device:      {device}")
    print(f"  Save Dir:    {SAVE_DIR}")
    print()

    # --- Step 1: Load checkpoints ---
    print("Step 1: Loading checkpoints...")

    gnnnet_path = f'checkpoints/gnnnet_{DATASET}.pt'
    h2gnn_path = f'checkpoints/h2gnn_{DATASET}.pt'
    data_path = f'checkpoints/data_{DATASET}.pt'

    if not all(os.path.exists(p) for p in [gnnnet_path, h2gnn_path, data_path]):
        print("  [ERROR] Checkpoints not found! Run main.py first to train and save.")
        print(f"  Expected: {gnnnet_path}, {h2gnn_path}, {data_path}")
        sys.exit(1)

    # Load data
    data_ckpt = torch.load(data_path, map_location=device)
    data_new = data_ckpt['data_new']
    nb_drugs = data_ckpt['nb_drugs']
    nb_proteins = data_ckpt['nb_proteins']
    nb_all = data_ckpt['nb_all']
    full_features = data_ckpt['features'].to(device)
    full_adj = data_ckpt['adj'].to(device)
    labels = data_ckpt['labels'].to(device)
    idx_test = data_ckpt['idx_test'].to(device)
    dataset = data_ckpt['dataset']

    print(f"  Loaded data: {nb_drugs} drugs, {nb_proteins} proteins, {nb_all} total nodes")
    print(f"  Features shape: {full_features.shape}")
    print(f"  Adj shape: {full_adj.shape}")

    # Load GNNNet
    args = opt.parser.parse_args([])
    gnnnet = GNNNet().to(device)
    gnnnet.load_state_dict(torch.load(gnnnet_path, map_location=device))
    gnnnet.eval()
    print(f"  Loaded GNNNet from {gnnnet_path}")

    # Load H2GNN
    h2gnn = H2GNN(n_node=nb_all).to(device)
    h2gnn.load_state_dict(torch.load(h2gnn_path, map_location=device))
    h2gnn.eval()
    print(f"  Loaded H2GNN from {h2gnn_path}")
    print()

    # --- Step 2: Build mappings and select TEST samples only ---
    print("Step 2: Selecting samples from the held-out test set...")
    drugmap, proteinmap, drugid_list, proteinid_list = build_drug_protein_maps(data_new)
    all_pairs = get_interaction_pairs(data_new, drugmap, proteinmap)
    selected_samples = select_test_pairs(
        all_pairs, idx_test, nb_drugs, nb_proteins, num_samples=NUM_SAMPLES, seed=42)
    print()

    # --- Step 3: Run explainability on each sample ---
    print("Step 3: Running explainability pipeline...")
    print()

    all_results = []
    for i, (d_idx, p_idx, d_id, p_id, label) in enumerate(selected_samples):
        try:
            result = run_single_sample(
                sample_idx=i,
                drug_idx=d_idx,
                protein_idx=p_idx,
                drug_id=d_id,
                protein_id=p_id,
                ground_truth=label,
                gnnnet=gnnnet,
                h2gnn=h2gnn,
                full_features=full_features,
                full_adj=full_adj,
                data_new=data_new,
                dataset=dataset,
                nb_drugs=nb_drugs,
                nb_all=nb_all,
                device=device,
                save_dir=SAVE_DIR,
                n_steps=N_STEPS,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] Sample {i} ({d_id}, {p_id}) failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # --- Step 4: Global analysis ---
    if all_results:
        run_global_analysis(NUM_SAMPLES, all_results, SAVE_DIR)

        # Save raw results
        results_path = os.path.join(SAVE_DIR, 'all_results.pt')
        torch.save(all_results, results_path)
        print(f"\n  Raw results saved to: {results_path}")

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

# End of low-level GAT explainability execution flow
