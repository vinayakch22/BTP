"""
=================================================================
PREDICTION-LEVEL EXPLAINABILITY FOR H2GnnDTI
=================================================================

Multi-level explainability pipeline that explains WHY the model
predicts a specific drug-target interaction.

Level 1: Attention Analysis (alpha, self-attention, GAT attention)
Level 2: Decoder Attribution (analytical gradient on z_hat)
Level 3: Fused Latent Attribution (Layer IG on z_tilde → prediction)
Level 4: AE Feature Attribution (isolated IG through AE MLP)
Level 5: IGAE Pair Attribution (isolated IG through IGAE GNN)

Method: Combination of attention extraction, analytical gradients,
        and Captum Integrated Gradients on isolated branches.
=================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

from captum.attr import IntegratedGradients

# Local imports
from model import H2GNN
import opt
import scipy.sparse as sp


# =====================================================================
# LEVEL 1: ATTENTION ANALYSIS
# =====================================================================

def level1_attention_analysis(h2gnn, features, adj, drug_idx, protein_idx,
                              nb_drugs, nb_all):
    """
    Extract all attention mechanisms from a single forward pass.
    No gradients needed — just tensor extraction.

    Returns:
        alpha_drug: (20,) — fusion gate weights for the drug node
        alpha_protein: (20,) — fusion gate weights for the protein node
        self_attn_score: scalar — how much drug attends to protein
        self_attn_drug_top: list — top nodes the drug attends to
        self_attn_protein_top: list — top nodes the protein attends to
    """
    h2gnn.eval()
    pro_feat_idx = nb_drugs + protein_idx

    with torch.no_grad():
        # Run forward pass to get intermediates
        x = features
        z_ae = h2gnn.ae.encoder(x)
        z_igae, z_igae_adj = h2gnn.gae.encoder(x, adj)

        combined_z = torch.cat((z_ae, z_igae), dim=1)
        alpha = h2gnn.fusion_gate(combined_z)

        z_i = alpha * z_ae + (1 - alpha) * z_igae

        edge_index = adj.nonzero().t().contiguous()

        # Get GAT attention weights
        # GATConv returns (out, (edge_index, alpha_weights)) when return_attention_weights=True
        z_i_gat, (gat_edge_index, gat_attn_weights) = h2gnn.gat_refine(
            z_i, edge_index, return_attention_weights=True
        )
        z_i = F.elu(z_i_gat)

        z_l = torch.spmm(adj, z_i)

        # Self-attention matrix
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)  # (nb_all, nb_all)

        # Decode
        z_g = torch.mm(s, z_l)
        z_tilde = h2gnn.gamma * z_g + z_l
        z_hat, z_hat_adj = h2gnn.gae.decoder(z_tilde, adj)

    # --- Extract alpha for drug and protein ---
    alpha_drug = alpha[drug_idx].cpu().numpy()       # (20,)
    alpha_protein = alpha[pro_feat_idx].cpu().numpy() # (20,)
    alpha_mean_drug = float(np.mean(alpha_drug))
    alpha_mean_protein = float(np.mean(alpha_protein))

    # --- Extract self-attention scores ---
    s_np = s.cpu().numpy()
    self_attn_drug_to_protein = float(s_np[drug_idx, pro_feat_idx])
    self_attn_protein_to_drug = float(s_np[pro_feat_idx, drug_idx])

    # Top-10 nodes the drug attends to
    drug_attn_row = s_np[drug_idx]
    top_drug_attn_idx = np.argsort(drug_attn_row)[::-1][:10]
    drug_top_attns = [(int(idx), float(drug_attn_row[idx]),
                       'drug' if idx < nb_drugs else 'protein')
                      for idx in top_drug_attn_idx]

    # Top-10 nodes the protein attends to
    pro_attn_row = s_np[pro_feat_idx]
    top_pro_attn_idx = np.argsort(pro_attn_row)[::-1][:10]
    pro_top_attns = [(int(idx), float(pro_attn_row[idx]),
                      'drug' if idx < nb_drugs else 'protein')
                     for idx in top_pro_attn_idx]

    # --- Extract GAT attention for edges involving drug_idx and protein_idx ---
    gat_ei = gat_edge_index.cpu().numpy()
    gat_aw = gat_attn_weights.cpu().numpy()  # (num_edges, num_heads)

    drug_gat_edges = []
    protein_gat_edges = []
    for e in range(gat_ei.shape[1]):
        src, dst = gat_ei[0, e], gat_ei[1, e]
        avg_attn = float(np.mean(gat_aw[e]))
        if src == drug_idx or dst == drug_idx:
            other = dst if src == drug_idx else src
            drug_gat_edges.append((other, avg_attn,
                                   'drug' if other < nb_drugs else 'protein'))
        if src == pro_feat_idx or dst == pro_feat_idx:
            other = dst if src == pro_feat_idx else src
            protein_gat_edges.append((other, avg_attn,
                                      'drug' if other < nb_drugs else 'protein'))

    drug_gat_edges.sort(key=lambda x: x[1], reverse=True)
    protein_gat_edges.sort(key=lambda x: x[1], reverse=True)

    return {
        'alpha_drug': alpha_drug,
        'alpha_protein': alpha_protein,
        'alpha_mean_drug': alpha_mean_drug,
        'alpha_mean_protein': alpha_mean_protein,
        'self_attn_drug_to_protein': self_attn_drug_to_protein,
        'self_attn_protein_to_drug': self_attn_protein_to_drug,
        'drug_top_self_attns': drug_top_attns,
        'protein_top_self_attns': pro_top_attns,
        'drug_gat_edges': drug_gat_edges[:10],
        'protein_gat_edges': protein_gat_edges[:10],
    }


# =====================================================================
# LEVEL 2: DECODER ATTRIBUTION (ANALYTICAL)
# =====================================================================

def level2_decoder_attribution(h2gnn, features, adj, drug_idx, protein_idx,
                               nb_drugs):
    """
    Compute analytical gradients of the prediction w.r.t. z_hat.

    pred = sigmoid(z_hat[i] · z_hat[j])
    ∂pred/∂z_hat[i] = sigmoid'(dot) × z_hat[j]
    ∂pred/∂z_hat[j] = sigmoid'(dot) × z_hat[i]

    Returns attribution over the 160-dim decoder output space.
    """
    h2gnn.eval()
    pro_feat_idx = nb_drugs + protein_idx

    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde, alpha = h2gnn(features, adj)

        # z_hat: (nb_all, 160) — IGAE decoder output
        z_hat_drug = z_hat[drug_idx]         # (160,)
        z_hat_protein = z_hat[pro_feat_idx]  # (160,)

        # Dot product (before sigmoid)
        dot_product = torch.dot(z_hat_drug, z_hat_protein)

        # Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        sig = torch.sigmoid(dot_product)
        sig_prime = sig * (1.0 - sig)

        # Analytical gradients
        grad_z_hat_drug = sig_prime * z_hat_protein    # (160,)
        grad_z_hat_protein = sig_prime * z_hat_drug    # (160,)

        # The attribution of each dimension = gradient × value (like IG with 1-step)
        attr_drug = (grad_z_hat_drug * z_hat_drug).cpu().numpy()      # (160,)
        attr_protein = (grad_z_hat_protein * z_hat_protein).cpu().numpy()  # (160,)

        # Also return raw gradients for dimension importance
        grad_drug_np = grad_z_hat_drug.cpu().numpy()
        grad_protein_np = grad_z_hat_protein.cpu().numpy()

    return {
        'pred_score': float(sig.item()),
        'dot_product': float(dot_product.item()),
        'sig_prime': float(sig_prime.item()),
        'attr_drug_dims': attr_drug,
        'attr_protein_dims': attr_protein,
        'grad_drug_dims': grad_drug_np,
        'grad_protein_dims': grad_protein_np,
        'z_hat_drug': z_hat_drug.cpu().numpy(),
        'z_hat_protein': z_hat_protein.cpu().numpy(),
    }


# =====================================================================
# LEVEL 3: FUSED LATENT ATTRIBUTION (Layer IG on z_tilde)
# =====================================================================

class H2GNNDecoderWrapper(nn.Module):
    """
    Wraps ONLY the decoder path: z_tilde → IGAE decoder → z_hat_adj[i,j]
    This avoids the self-attention bottleneck.
    """
    def __init__(self, igae_decoder, adj, drug_idx, protein_idx, nb_drugs):
        super().__init__()
        self.igae_decoder = igae_decoder
        self.adj = adj
        self.drug_idx = drug_idx
        self.pro_feat_idx = nb_drugs + protein_idx

    def forward(self, z_tilde):
        # z_tilde: (S, 2293, 20)
        outputs = []
        for s in range(z_tilde.size(0)):
            z_s = z_tilde[s]  # (2293, 20)
            z_hat, z_hat_adj = self.igae_decoder(z_s, self.adj)
            pred = z_hat_adj[self.drug_idx, self.pro_feat_idx]
            outputs.append(pred)
        return torch.stack(outputs)  # (S,)


def level3_fused_latent_attribution(h2gnn, features, adj, drug_idx, protein_idx,
                                     nb_drugs, n_steps=200):
    """
    Attribute the prediction to z_tilde dimensions using Integrated Gradients.
    Path: z_tilde → IGAE decoder (3 GNN layers) → z_hat_adj[i,j]
    """
    h2gnn.eval()
    pro_feat_idx = nb_drugs + protein_idx

    # First get z_tilde from a normal forward pass
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde_orig, alpha = h2gnn(features, adj)

    # Create decoder-only wrapper
    wrapper = H2GNNDecoderWrapper(
        h2gnn.gae.decoder, adj, drug_idx, protein_idx, nb_drugs
    )

    # Make z_tilde require grad for IG, and add batch dimension of 1
    z_tilde_input = z_tilde_orig.unsqueeze(0).clone().detach().requires_grad_(True)
    baseline = torch.zeros_like(z_tilde_input)

    ig = IntegratedGradients(wrapper)
    attributions = ig.attribute(
        z_tilde_input,
        baselines=baseline,
        n_steps=n_steps,
        method='gausslegendre',
        return_convergence_delta=False
    )

    # Extract attributions for drug and protein rows from batch 0
    attr_drug = attributions[0, drug_idx].detach().cpu().numpy()       # (20,)
    attr_protein = attributions[0, pro_feat_idx].detach().cpu().numpy() # (20,)

    # Node-level importance (sum of absolute dim attributions)
    all_node_imp = torch.sum(torch.abs(attributions[0]), dim=1).detach().cpu().numpy()

    return {
        'attr_z_tilde_drug': attr_drug,
        'attr_z_tilde_protein': attr_protein,
        'all_node_importance': all_node_imp,
        'nonzero_count': int(np.count_nonzero(attributions[0].detach().cpu().numpy())),
    }


# =====================================================================
# LEVEL 4: AE FEATURE ATTRIBUTION (Isolated IG)
# =====================================================================

class AEPairWrapper(nn.Module):
    """
    Wraps the AE encoder for a drug-protein pair.
    
    Input: features (nb_all, 160)
    Path:  features[i] → AE encoder → z_ae[i]
           features[j] → AE encoder → z_ae[j]
           output = z_ae[i] · z_ae[j]  (dot product as surrogate prediction)
    """
    def __init__(self, ae_encoder, drug_idx, protein_idx, nb_drugs):
        super().__init__()
        self.ae_encoder = ae_encoder
        self.drug_idx = drug_idx
        self.pro_feat_idx = nb_drugs + protein_idx

    def forward(self, features):
        # features: (S, 2293, 160)
        z_ae = self.ae_encoder(features)  # (S, 2293, 20)
        z_ae_drug = z_ae[:, self.drug_idx]  # (S, 20)
        z_ae_protein = z_ae[:, self.pro_feat_idx]  # (S, 20)
        score = torch.sum(z_ae_drug * z_ae_protein, dim=1)  # (S,)
        return score


def level4_ae_feature_attribution(h2gnn, features, adj, drug_idx, protein_idx,
                                   nb_drugs, n_steps=200):
    """
    Attribute the AE-based surrogate prediction to input features.
    Path: features → AE encoder (pure MLP) → z_ae[i] · z_ae[j]
    """
    h2gnn.eval()
    pro_feat_idx = nb_drugs + protein_idx

    wrapper = AEPairWrapper(h2gnn.ae.encoder, drug_idx, protein_idx, nb_drugs)

    # Add batch dimension of 1
    features_input = features.unsqueeze(0).clone().detach().requires_grad_(True)
    baseline = torch.zeros_like(features_input)

    ig = IntegratedGradients(wrapper)
    attributions = ig.attribute(
        features_input,
        baselines=baseline,
        n_steps=n_steps,
        method='gausslegendre',
        return_convergence_delta=False
    )

    attr_drug = attributions[0, drug_idx].detach().cpu().numpy()        # (160,)
    attr_protein = attributions[0, pro_feat_idx].detach().cpu().numpy()  # (160,)

    # Feature importance across all nodes
    all_node_imp = torch.sum(torch.abs(attributions[0]), dim=1).detach().cpu().numpy()

    return {
        'attr_features_drug': attr_drug,
        'attr_features_protein': attr_protein,
        'drug_top_features': np.argsort(np.abs(attr_drug))[::-1][:20].tolist(),
        'protein_top_features': np.argsort(np.abs(attr_protein))[::-1][:20].tolist(),
        'all_node_importance': all_node_imp,
        'nonzero_drug': int(np.count_nonzero(attr_drug)),
        'nonzero_protein': int(np.count_nonzero(attr_protein)),
    }


# =====================================================================
# LEVEL 5: IGAE PAIR ATTRIBUTION (Isolated IG)
# =====================================================================

class IGAEPairWrapper(nn.Module):
    """
    Wraps the IGAE encoder for a drug-protein pair.
    
    Input: features (nb_all, 160)
    Path:  features → IGAE encoder (3 GNN layers with adj) → z_igae
           z_igae_adj = sigmoid(z_igae @ z_igae.T)
           output = z_igae_adj[i, j]
    """
    def __init__(self, igae_encoder, adj, drug_idx, protein_idx, nb_drugs):
        super().__init__()
        self.igae_encoder = igae_encoder
        self.adj = adj
        self.drug_idx = drug_idx
        self.pro_feat_idx = nb_drugs + protein_idx

    def forward(self, features):
        # features: (S, 2293, 160)
        outputs = []
        for s in range(features.size(0)):
            feat_s = features[s]  # (2293, 160)
            z_igae, z_igae_adj = self.igae_encoder(feat_s, self.adj)
            pred = z_igae_adj[self.drug_idx, self.pro_feat_idx]
            outputs.append(pred)
        return torch.stack(outputs)  # (S,)


def level5_igae_pair_attribution(h2gnn, features, adj, drug_idx, protein_idx,
                                  nb_drugs, n_steps=200):
    """
    Attribute the IGAE reconstruction score to input features.
    Path: features → IGAE encoder (3 GNN layers) → z_igae_adj[i, j]
    """
    h2gnn.eval()
    pro_feat_idx = nb_drugs + protein_idx

    wrapper = IGAEPairWrapper(h2gnn.gae.encoder, adj, drug_idx, protein_idx, nb_drugs)

    # Add batch dimension of 1
    features_input = features.unsqueeze(0).clone().detach().requires_grad_(True)
    baseline = torch.zeros_like(features_input)

    ig = IntegratedGradients(wrapper)
    attributions = ig.attribute(
        features_input,
        baselines=baseline,
        n_steps=n_steps,
        method='gausslegendre',
        return_convergence_delta=False
    )

    attr_drug = attributions[0, drug_idx].detach().cpu().numpy()        # (160,)
    attr_protein = attributions[0, pro_feat_idx].detach().cpu().numpy()  # (160,)

    all_node_imp = torch.sum(torch.abs(attributions[0]), dim=1).detach().cpu().numpy()

    return {
        'attr_features_drug': attr_drug,
        'attr_features_protein': attr_protein,
        'drug_top_features': np.argsort(np.abs(attr_drug))[::-1][:20].tolist(),
        'protein_top_features': np.argsort(np.abs(attr_protein))[::-1][:20].tolist(),
        'all_node_importance': all_node_imp,
        'nonzero_drug': int(np.count_nonzero(attr_drug)),
        'nonzero_protein': int(np.count_nonzero(attr_protein)),
    }


# =====================================================================
# VISUALIZATION
# =====================================================================

def visualize_sample(sample_id, drug_id, protein_id, ground_truth,
                     l1_results, l2_results, l3_results, l4_results, l5_results,
                     nb_drugs, save_dir):
    """Generate multi-panel visualization for one sample across all levels."""

    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle(f'Sample {sample_id}: Drug {drug_id} — Protein {protein_id}\n'
                 f'GT={ground_truth}, Pred={l2_results["pred_score"]:.4f}',
                 fontsize=16, fontweight='bold')

    # --- Panel 1: Alpha Fusion (Level 1) ---
    ax = axes[0, 0]
    x_pos = np.arange(20)
    ax.bar(x_pos - 0.15, l1_results['alpha_drug'], 0.3, label='Drug (AE weight)',
           color='steelblue', alpha=0.8)
    ax.bar(x_pos + 0.15, l1_results['alpha_protein'], 0.3, label='Protein (AE weight)',
           color='coral', alpha=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Alpha (AE Weight)')
    ax.set_title(f'L1: Dynamic Fusion Gate\n'
                 f'Drug ᾱ={l1_results["alpha_mean_drug"]:.3f}, '
                 f'Protein ᾱ={l1_results["alpha_mean_protein"]:.3f}')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # --- Panel 2: z_hat Attribution (Level 2) ---
    ax = axes[0, 1]
    top_k = 30
    drug_attr = l2_results['attr_drug_dims']
    protein_attr = l2_results['attr_protein_dims']
    # Show top dimensions by absolute attribution
    combined = np.abs(drug_attr) + np.abs(protein_attr)
    top_dims = np.argsort(combined)[::-1][:top_k]
    ax.bar(range(top_k), drug_attr[top_dims], 0.4, label='Drug z_hat attr',
           color='steelblue', alpha=0.8)
    ax.bar([x + 0.4 for x in range(top_k)], protein_attr[top_dims], 0.4,
           label='Protein z_hat attr', color='coral', alpha=0.8)
    ax.set_xlabel(f'Top {top_k} Latent Dimensions (of 160)')
    ax.set_ylabel('Attribution (gradient × value)')
    ax.set_title(f'L2: Decoder Attribution (Analytical)\n'
                 f'σ\'={l2_results["sig_prime"]:.6f}')
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, top_k, 5))
    ax.set_xticklabels([str(top_dims[i]) for i in range(0, top_k, 5)], fontsize=7)

    # --- Panel 3: z_tilde Attribution (Level 3) ---
    ax = axes[1, 0]
    z_tilde_drug = l3_results['attr_z_tilde_drug']
    z_tilde_protein = l3_results['attr_z_tilde_protein']
    ax.bar(np.arange(20) - 0.15, z_tilde_drug, 0.3, label='Drug z_tilde',
           color='steelblue', alpha=0.8)
    ax.bar(np.arange(20) + 0.15, z_tilde_protein, 0.3, label='Protein z_tilde',
           color='coral', alpha=0.8)
    ax.set_xlabel('Fused Latent Dimension (n_z=20)')
    ax.set_ylabel('IG Attribution')
    ax.set_title(f'L3: Fused Latent Attribution (Layer IG)\n'
                 f'Nonzero: {l3_results["nonzero_count"]}')
    ax.legend(fontsize=8)

    # --- Panel 4: AE Feature Attribution (Level 4) ---
    ax = axes[1, 1]
    ae_drug = l4_results['attr_features_drug']
    ae_protein = l4_results['attr_features_protein']
    top_feat = 30
    combined_ae = np.abs(ae_drug) + np.abs(ae_protein)
    top_ae_dims = np.argsort(combined_ae)[::-1][:top_feat]
    ax.bar(range(top_feat), ae_drug[top_ae_dims], 0.4, label='Drug features',
           color='steelblue', alpha=0.8)
    ax.bar([x + 0.4 for x in range(top_feat)], ae_protein[top_ae_dims], 0.4,
           label='Protein features', color='coral', alpha=0.8)
    ax.set_xlabel(f'Top {top_feat} Feature Dimensions (of 160)')
    ax.set_ylabel('IG Attribution')
    ax.set_title(f'L4: AE Feature Attribution\n'
                 f'Drug nonzero: {l4_results["nonzero_drug"]}, '
                 f'Protein nonzero: {l4_results["nonzero_protein"]}')
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, top_feat, 5))
    ax.set_xticklabels([str(top_ae_dims[i]) for i in range(0, top_feat, 5)], fontsize=7)

    # --- Panel 5: IGAE Feature Attribution (Level 5) ---
    ax = axes[2, 0]
    igae_drug = l5_results['attr_features_drug']
    igae_protein = l5_results['attr_features_protein']
    combined_igae = np.abs(igae_drug) + np.abs(igae_protein)
    top_igae_dims = np.argsort(combined_igae)[::-1][:top_feat]
    ax.bar(range(top_feat), igae_drug[top_igae_dims], 0.4, label='Drug features',
           color='steelblue', alpha=0.8)
    ax.bar([x + 0.4 for x in range(top_feat)], igae_protein[top_igae_dims], 0.4,
           label='Protein features', color='coral', alpha=0.8)
    ax.set_xlabel(f'Top {top_feat} Feature Dimensions (of 160)')
    ax.set_ylabel('IG Attribution')
    ax.set_title(f'L5: IGAE Feature Attribution\n'
                 f'Drug nonzero: {l5_results["nonzero_drug"]}, '
                 f'Protein nonzero: {l5_results["nonzero_protein"]}')
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, top_feat, 5))
    ax.set_xticklabels([str(top_igae_dims[i]) for i in range(0, top_feat, 5)], fontsize=7)

    # --- Panel 6: AE vs IGAE comparison ---
    ax = axes[2, 1]
    ae_drug_imp = np.sum(np.abs(ae_drug))
    ae_pro_imp = np.sum(np.abs(ae_protein))
    igae_drug_imp = np.sum(np.abs(igae_drug))
    igae_pro_imp = np.sum(np.abs(igae_protein))
    categories = ['AE\nDrug', 'AE\nProtein', 'IGAE\nDrug', 'IGAE\nProtein']
    values = [ae_drug_imp, ae_pro_imp, igae_drug_imp, igae_pro_imp]
    colors = ['steelblue', 'coral', 'navy', 'darkred']
    ax.bar(categories, values, color=colors, alpha=0.8)
    ax.set_ylabel('Total Absolute Attribution')
    ax.set_title('AE vs IGAE: Total Feature Importance')

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f'sample_{sample_id}_prediction_explainability.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig_path


# =====================================================================
# SINGLE SAMPLE PIPELINE
# =====================================================================

def run_single_sample(sample_idx, drug_idx, protein_idx, drug_id, protein_id,
                      ground_truth, h2gnn, features, adj, nb_drugs, nb_all,
                      device, save_dir, n_steps=200):
    """
    Run all 5 levels of prediction-level explainability for one pair.
    """
    pro_feat_idx = nb_drugs + protein_idx

    print("=" * 70)
    print("PREDICTION-LEVEL EXPLAINABILITY")
    print("=" * 70)
    print(f"  Sample:       {sample_idx}")
    print(f"  Drug:         {drug_id} (idx={drug_idx})")
    print(f"  Protein:      {protein_id} (idx={protein_idx}, feat_row={pro_feat_idx})")
    print(f"  Ground Truth: {ground_truth}")
    print()

    # Get overall prediction
    h2gnn.eval()
    with torch.no_grad():
        _, _, adj_hat, _, _, _, _ = h2gnn(features, adj)
        pred_raw = adj_hat[drug_idx, pro_feat_idx].item()
    print(f"  Prediction:   {pred_raw:.4f} (from adj_hat, already sigmoid)")
    print()

    # ==================== LEVEL 1 ====================
    print("  -- Level 1: Attention Analysis --")
    l1 = level1_attention_analysis(h2gnn, features, adj, drug_idx, protein_idx,
                                   nb_drugs, nb_all)

    print(f"    Alpha (Drug):    mean={l1['alpha_mean_drug']:.4f}  "
          f"({'AE-dominant' if l1['alpha_mean_drug'] > 0.5 else 'IGAE-dominant'})")
    print(f"    Alpha (Protein): mean={l1['alpha_mean_protein']:.4f}  "
          f"({'AE-dominant' if l1['alpha_mean_protein'] > 0.5 else 'IGAE-dominant'})")
    print(f"    Self-Attn Drug->Protein: {l1['self_attn_drug_to_protein']:.6f}")
    print(f"    Self-Attn Protein->Drug: {l1['self_attn_protein_to_drug']:.6f}")
    print(f"    Drug's top self-attention targets:")
    for idx, score, ntype in l1['drug_top_self_attns'][:5]:
        marker = " <<<" if idx == pro_feat_idx else ""
        print(f"      Node {idx:4d} ({ntype:7s}): {score:.6f}{marker}")
    print(f"    Drug's top GAT neighbors:")
    for idx, score, ntype in l1['drug_gat_edges'][:5]:
        marker = " <<<" if idx == pro_feat_idx else ""
        print(f"      Node {idx:4d} ({ntype:7s}): {score:.6f}{marker}")
    print()

    # ==================== LEVEL 2 ====================
    print("  -- Level 2: Decoder Attribution (Analytical) --")
    l2 = level2_decoder_attribution(h2gnn, features, adj, drug_idx, protein_idx,
                                     nb_drugs)

    print(f"    Prediction (sigmoid): {l2['pred_score']:.4f}")
    print(f"    Dot product (pre-sigmoid): {l2['dot_product']:.4f}")
    print(f"    Sigmoid derivative: {l2['sig_prime']:.6f}")

    drug_top = np.argsort(np.abs(l2['attr_drug_dims']))[::-1][:5]
    pro_top = np.argsort(np.abs(l2['attr_protein_dims']))[::-1][:5]
    drug_attr_vals = [f'{l2["attr_drug_dims"][d]:.6f}' for d in drug_top]
    pro_attr_vals = [f'{l2["attr_protein_dims"][d]:.6f}' for d in pro_top]
    print(f"    Top drug z_hat dims:    {drug_top.tolist()} "
          f"(attrs: {drug_attr_vals})")
    print(f"    Top protein z_hat dims: {pro_top.tolist()} "
          f"(attrs: {pro_attr_vals})")
    print()

    # ==================== LEVEL 3 ====================
    print(f"  -- Level 3: Fused Latent Attribution (Layer IG, {n_steps} steps) --")
    l3 = level3_fused_latent_attribution(h2gnn, features, adj, drug_idx,
                                          protein_idx, nb_drugs, n_steps)

    print(f"    Nonzero attributions: {l3['nonzero_count']}")
    drug_zt = l3['attr_z_tilde_drug']
    pro_zt = l3['attr_z_tilde_protein']
    drug_top_zt = np.argsort(np.abs(drug_zt))[::-1][:5]
    pro_top_zt = np.argsort(np.abs(pro_zt))[::-1][:5]
    print(f"    Drug z_tilde top dims:    {drug_top_zt.tolist()} "
          f"(attrs: {[f'{drug_zt[d]:.6f}' for d in drug_top_zt]})")
    print(f"    Protein z_tilde top dims: {pro_top_zt.tolist()} "
          f"(attrs: {[f'{pro_zt[d]:.6f}' for d in pro_top_zt]})")
    print()

    # ==================== LEVEL 4 ====================
    print(f"  -- Level 4: AE Feature Attribution (IG, {n_steps} steps) --")
    l4 = level4_ae_feature_attribution(h2gnn, features, adj, drug_idx,
                                        protein_idx, nb_drugs, n_steps)

    print(f"    Drug:    {l4['nonzero_drug']}/160 nonzero features")
    print(f"    Protein: {l4['nonzero_protein']}/160 nonzero features")
    print(f"    Drug top feature dims:    {l4['drug_top_features'][:10]}")
    print(f"    Protein top feature dims: {l4['protein_top_features'][:10]}")
    print()

    # ==================== LEVEL 5 ====================
    print(f"  -- Level 5: IGAE Pair Attribution (IG, {n_steps} steps) --")
    l5 = level5_igae_pair_attribution(h2gnn, features, adj, drug_idx,
                                       protein_idx, nb_drugs, n_steps)

    print(f"    Drug:    {l5['nonzero_drug']}/160 nonzero features")
    print(f"    Protein: {l5['nonzero_protein']}/160 nonzero features")
    print(f"    Drug top feature dims:    {l5['drug_top_features'][:10]}")
    print(f"    Protein top feature dims: {l5['protein_top_features'][:10]}")
    print()

    # ==================== VISUALIZATION ====================
    print("  Generating visualization...")
    fig_path = visualize_sample(sample_idx, drug_id, protein_id, ground_truth,
                                l1, l2, l3, l4, l5, nb_drugs, save_dir)
    print(f"    Saved: {fig_path}")
    print()
    print("=" * 70)
    print()

    return {
        'sample_idx': sample_idx,
        'drug_id': drug_id,
        'protein_id': protein_id,
        'ground_truth': ground_truth,
        'prediction': pred_raw,
        'level1': l1,
        'level2': l2,
        'level3': l3,
        'level4': l4,
        'level5': l5,
    }


# =====================================================================
# GLOBAL ANALYSIS
# =====================================================================

def run_global_analysis(all_results, save_dir):
    """Aggregate results across all samples."""
    print()
    print("=" * 70)
    print("GLOBAL ANALYSIS (PREDICTION-LEVEL)")
    print("=" * 70)
    n = len(all_results)
    print(f"  Analyzed {n} samples")

    # --- Alpha statistics ---
    alpha_drugs = [r['level1']['alpha_mean_drug'] for r in all_results]
    alpha_prots = [r['level1']['alpha_mean_protein'] for r in all_results]
    pos_results = [r for r in all_results if r['ground_truth'] == 1]
    neg_results = [r for r in all_results if r['ground_truth'] == 0]

    print(f"\n  Level 1: Alpha Fusion Gate Statistics")
    print(f"    Drug alpha:    mean={np.mean(alpha_drugs):.4f}, std={np.std(alpha_drugs):.4f}")
    print(f"    Protein alpha: mean={np.mean(alpha_prots):.4f}, std={np.std(alpha_prots):.4f}")
    if pos_results and neg_results:
        pos_alpha = np.mean([r['level1']['alpha_mean_drug'] for r in pos_results])
        neg_alpha = np.mean([r['level1']['alpha_mean_drug'] for r in neg_results])
        print(f"    Positive pairs drug alpha: {pos_alpha:.4f}")
        print(f"    Negative pairs drug alpha: {neg_alpha:.4f}")

    # --- Self-attention statistics ---
    d2p_scores = [r['level1']['self_attn_drug_to_protein'] for r in all_results]
    print(f"\n  Level 1: Self-Attention Drug->Protein")
    print(f"    mean={np.mean(d2p_scores):.6f}, max={np.max(d2p_scores):.6f}")

    # --- Level 2 statistics ---
    sig_primes = [r['level2']['sig_prime'] for r in all_results]
    print(f"\n  Level 2: Sigmoid Derivative Distribution")
    print(f"    mean={np.mean(sig_primes):.6f}, std={np.std(sig_primes):.6f}")

    # --- Level 3 statistics ---
    l3_nonzero = [r['level3']['nonzero_count'] for r in all_results]
    print(f"\n  Level 3: Fused Latent Attribution")
    print(f"    Mean nonzero attributions: {np.mean(l3_nonzero):.0f}")

    # --- Level 4 vs 5 comparison ---
    ae_drug_imp = [np.sum(np.abs(r['level4']['attr_features_drug'])) for r in all_results]
    igae_drug_imp = [np.sum(np.abs(r['level5']['attr_features_drug'])) for r in all_results]
    print(f"\n  Level 4 vs 5: AE vs IGAE Total Feature Attribution")
    print(f"    AE drug total:   mean={np.mean(ae_drug_imp):.6f}")
    print(f"    IGAE drug total: mean={np.mean(igae_drug_imp):.6f}")
    ratio = np.mean(igae_drug_imp) / (np.mean(ae_drug_imp) + 1e-10)
    print(f"    IGAE/AE ratio:   {ratio:.2f}x")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Alpha distribution
    axes[0, 0].hist(alpha_drugs, bins=20, alpha=0.6, label='Drugs', color='steelblue')
    axes[0, 0].hist(alpha_prots, bins=20, alpha=0.6, label='Proteins', color='coral')
    axes[0, 0].axvline(x=0.5, color='black', linestyle='--')
    axes[0, 0].set_title('Alpha Fusion Gate Distribution')
    axes[0, 0].set_xlabel('Mean Alpha (higher = more AE)')
    axes[0, 0].legend()

    # Predictions: positive vs negative
    preds = [r['prediction'] for r in all_results]
    gts = [r['ground_truth'] for r in all_results]
    colors = ['green' if gt == 1 else 'red' for gt in gts]
    axes[0, 1].scatter(range(n), preds, c=colors, alpha=0.6, s=40)
    axes[0, 1].axhline(y=0.5, color='black', linestyle='--')
    axes[0, 1].set_title('Predictions by Ground Truth')
    axes[0, 1].set_ylabel('Prediction Score')
    axes[0, 1].legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Positive'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Negative')
    ])

    # AE vs IGAE attribution
    axes[1, 0].scatter(ae_drug_imp, igae_drug_imp, c=colors, alpha=0.6, s=40)
    axes[1, 0].set_xlabel('AE Total Attribution (Drug)')
    axes[1, 0].set_ylabel('IGAE Total Attribution (Drug)')
    axes[1, 0].set_title('AE vs IGAE Feature Attribution')
    max_val = max(max(ae_drug_imp), max(igae_drug_imp)) * 1.1
    axes[1, 0].plot([0, max_val], [0, max_val], 'k--', alpha=0.3)

    # Self-attention drug→protein scores
    pos_d2p = [r['level1']['self_attn_drug_to_protein'] for r in pos_results]
    neg_d2p = [r['level1']['self_attn_drug_to_protein'] for r in neg_results]
    if pos_d2p and neg_d2p:
        axes[1, 1].hist(pos_d2p, bins=15, alpha=0.6, label='Positive', color='green')
        axes[1, 1].hist(neg_d2p, bins=15, alpha=0.6, label='Negative', color='red')
        axes[1, 1].set_title('Self-Attention Drug→Protein by GT')
        axes[1, 1].set_xlabel('Attention Score')
        axes[1, 1].legend()

    plt.suptitle('Global Prediction-Level Explainability', fontsize=16)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'global_prediction_explainability.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {fig_path}")
    print("=" * 70)


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    print()
    print("=" * 70)
    print("H2GnnDTI PREDICTION-LEVEL EXPLAINABILITY PIPELINE")
    print("=" * 70)

    DATASET = "kiba"
    NUM_SAMPLES = 100
    N_STEPS = 50
    SAVE_DIR = os.path.join('results', 'explainability_highlevel')
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device('cpu')
    print(f"  Dataset:  {DATASET}")
    print(f"  Samples:  {NUM_SAMPLES}")
    print(f"  IG Steps: {N_STEPS}")
    print(f"  Save Dir: {SAVE_DIR}")
    print()

    # --- Load checkpoints ---
    print("Loading checkpoints...")
    data_ckpt = torch.load(f'checkpoints/data_{DATASET}.pt', map_location=device)
    data_new = data_ckpt['data_new']
    nb_drugs = data_ckpt['nb_drugs']
    nb_proteins = data_ckpt['nb_proteins']
    nb_all = data_ckpt['nb_all']
    features = data_ckpt['features'].to(device)
    adj = data_ckpt['adj'].to(device)
    labels = data_ckpt['labels'].to(device)
    idx_test = data_ckpt['idx_test']
    print(f"  {nb_drugs} drugs, {nb_proteins} proteins, {nb_all} total")

    args = opt.parser.parse_args([])
    h2gnn = H2GNN(n_node=nb_all).to(device)
    h2gnn.load_state_dict(torch.load(f'checkpoints/h2gnn_{DATASET}.pt', map_location=device))
    h2gnn.eval()
    print("  H2GNN loaded")
    print()

    # --- Build sample list from the held-out test set only ---
    from explainability_lowlevel import build_drug_protein_maps, get_interaction_pairs, select_test_pairs
    drugmap, proteinmap, _, _ = build_drug_protein_maps(data_new)
    all_pairs = get_interaction_pairs(data_new, drugmap, proteinmap)
    selected = select_test_pairs(
        all_pairs, idx_test, nb_drugs, nb_proteins, num_samples=NUM_SAMPLES, seed=42)
    print()

    # --- Run pipeline ---
    all_results = []
    for i, (d_idx, p_idx, d_id, p_id, label) in enumerate(selected):
        try:
            result = run_single_sample(
                sample_idx=i, drug_idx=d_idx, protein_idx=p_idx,
                drug_id=d_id, protein_id=p_id, ground_truth=label,
                h2gnn=h2gnn, features=features, adj=adj,
                nb_drugs=nb_drugs, nb_all=nb_all, device=device,
                save_dir=SAVE_DIR, n_steps=N_STEPS,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] Sample {i} ({d_id}, {p_id}): {e}")
            import traceback
            traceback.print_exc()

    # --- Global analysis ---
    if all_results:
        run_global_analysis(all_results, SAVE_DIR)
        torch.save(all_results, os.path.join(SAVE_DIR, 'all_results.pt'))
        print(f"\n  Raw results saved to: {SAVE_DIR}/all_results.pt")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()

# End of prediction-level explainability pipeline
