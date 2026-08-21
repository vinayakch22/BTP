"""
=================================================================
VALIDATION: EDGE IMPORTANCE SANITY CHECK (LL-6)
=================================================================

OBJECTIVE:
  Validate that the edge (bond/contact) importance scores derived
  from node importance are internally consistent and chemically
  meaningful.

WHY THIS MATTERS:
  In our pipeline, edge importance is computed as:
      edge_imp(i->j) = node_imp[i] + node_imp[j]

  This is an approximation. LL-6 checks whether this derivation
  produces sensible results by testing three properties:

  1. CONSISTENCY: Edge importance should strongly correlate with
     the endpoint node importances (Pearson r should be high).

  2. BOND-TYPE DISCRIMINATION (Drugs): Important bonds should
     connect specific atom types (N, O, S) more often than random
     bonds. We check if top-ranked bonds preferentially connect
     heteroatom-containing endpoints.

  3. DEGREE BIAS CHECK: We check whether the edge importance is
     simply a proxy for node degree (high-degree nodes appearing
     important just because they have many connections, not
     because they are chemically meaningful). If importance is
     independent of degree, the model learned genuine structure.

INPUTS:
  - Saved node importance scores from all_results.pt
  - Drug SMILES (from data checkpoint) to reconstruct graphs
  - Protein graphs (from data checkpoint)

OUTPUTS:
  - Pearson correlation between edge importance and endpoint
    node importance (mean, max endpoint)
  - Bond-type analysis: fraction of top bonds connecting
    heteroatoms vs random bonds
  - Degree-importance correlation analysis
  - Figures and statistical report
Re-run explainability after the leakage-fixed retraining; old all_results.pt files are invalid.
=================================================================
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from collections import defaultdict

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from setting import smile_to_graph
from explainability_lowlevel import (
    build_drug_protein_maps,
    get_protein_graph_data,
)


# =====================================================================
# HELPER: Recompute edge importance from node importance
# =====================================================================

def compute_edge_importance(node_imp, edge_index):
    """
    Recompute edge importance as sum of endpoint node importances.
    Returns edge_imp array and list of (src, dst, importance).
    """
    src = edge_index[0]
    dst = edge_index[1]
    edge_imp = node_imp[src] + node_imp[dst]

    # Deduplicated sorted edges
    seen = set()
    top_edges = []
    sorted_idx = np.argsort(edge_imp)[::-1]
    for idx in sorted_idx:
        s, d = int(src[idx]), int(dst[idx])
        if s == d:
            continue
        key = (min(s, d), max(s, d))
        if key not in seen:
            seen.add(key)
            top_edges.append((s, d, edge_imp[idx]))
    return edge_imp, top_edges


# =====================================================================
# TEST 1: CONSISTENCY — Edge imp vs Endpoint node imp
# =====================================================================

def test_consistency(ll_results, data_new, drug_smiles_map, dataset, save_dir):
    """
    For each sample, check Pearson r between edge importance and
    the mean/max of endpoint node importances.
    """
    print("=" * 70)
    print("TEST 1: EDGE-NODE CONSISTENCY (Pearson Correlation)")
    print("=" * 70)
    print()

    drug_pearson_mean = []  # r between edge_imp and mean(node[src], node[dst])
    drug_pearson_max = []   # r between edge_imp and max(node[src], node[dst])
    pro_pearson_mean = []
    pro_pearson_max = []

    for s_idx, result in enumerate(ll_results):
        drug_id = result['drug_id']
        protein_id = result['protein_id']
        drug_imp = result['drug_node_importance']
        pro_imp = result['protein_node_importance']
        if isinstance(drug_imp, torch.Tensor):
            drug_imp = drug_imp.numpy()
        if isinstance(pro_imp, torch.Tensor):
            pro_imp = pro_imp.numpy()

        # --- Drug ---
        smiles = drug_smiles_map.get(drug_id)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                c_size, features, edge_list = smile_to_graph(smiles)
                ei = np.array(edge_list).T  # (2, num_edges)
                if ei.shape[1] > 2:
                    src, dst = ei[0], ei[1]
                    # Filter to valid atoms
                    mask = (src < len(drug_imp)) & (dst < len(drug_imp))
                    src, dst = src[mask], dst[mask]
                    if len(src) > 2:
                        edge_imp = drug_imp[src] + drug_imp[dst]
                        mean_endpoint = (drug_imp[src] + drug_imp[dst]) / 2
                        max_endpoint = np.maximum(drug_imp[src], drug_imp[dst])

                        r_mean, _ = pearsonr(edge_imp, mean_endpoint)
                        r_max, _ = pearsonr(edge_imp, max_endpoint)
                        drug_pearson_mean.append(r_mean)
                        drug_pearson_max.append(r_max)

        # --- Protein ---
        try:
            pro_data, _ = get_protein_graph_data(protein_id, data_new, dataset)
            ei_pro = pro_data.edge_index.numpy()
            src_p, dst_p = ei_pro[0], ei_pro[1]
            mask_p = (src_p < len(pro_imp)) & (dst_p < len(pro_imp))
            src_p, dst_p = src_p[mask_p], dst_p[mask_p]
            if len(src_p) > 2:
                edge_imp_p = pro_imp[src_p] + pro_imp[dst_p]
                mean_ep = (pro_imp[src_p] + pro_imp[dst_p]) / 2
                max_ep = np.maximum(pro_imp[src_p], pro_imp[dst_p])

                r_mean_p, _ = pearsonr(edge_imp_p, mean_ep)
                r_max_p, _ = pearsonr(edge_imp_p, max_ep)
                pro_pearson_mean.append(r_mean_p)
                pro_pearson_max.append(r_max_p)
        except Exception:
            pass

    print(f"  Drug edge-node Pearson (mean endpoint): "
          f"mean={np.mean(drug_pearson_mean):.4f}, std={np.std(drug_pearson_mean):.4f}")
    print(f"  Drug edge-node Pearson (max endpoint):  "
          f"mean={np.mean(drug_pearson_max):.4f}, std={np.std(drug_pearson_max):.4f}")
    print(f"  Prot edge-node Pearson (mean endpoint): "
          f"mean={np.mean(pro_pearson_mean):.4f}, std={np.std(pro_pearson_mean):.4f}")
    print(f"  Prot edge-node Pearson (max endpoint):  "
          f"mean={np.mean(pro_pearson_max):.4f}, std={np.std(pro_pearson_max):.4f}")
    print()

    return {
        'drug_pearson_mean': drug_pearson_mean,
        'drug_pearson_max': drug_pearson_max,
        'pro_pearson_mean': pro_pearson_mean,
        'pro_pearson_max': pro_pearson_max,
    }


# =====================================================================
# TEST 2: BOND-TYPE DISCRIMINATION (Drugs only)
# =====================================================================

def test_bond_type_discrimination(ll_results, drug_smiles_map, save_dir):
    """
    Check if top-ranked bonds preferentially connect heteroatoms
    compared to random bonds.
    """
    print("=" * 70)
    print("TEST 2: BOND-TYPE DISCRIMINATION")
    print("=" * 70)
    print()

    top10_hetero_fracs = []
    random10_hetero_fracs = []
    top10_aromatic_fracs = []
    random10_aromatic_fracs = []

    np.random.seed(42)

    for s_idx, result in enumerate(ll_results):
        drug_id = result['drug_id']
        drug_imp = result['drug_node_importance']
        if isinstance(drug_imp, torch.Tensor):
            drug_imp = drug_imp.numpy()

        smiles = drug_smiles_map.get(drug_id)
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue

        num_atoms = mol.GetNumAtoms()
        c_size, features, edge_list = smile_to_graph(smiles)
        ei = np.array(edge_list).T

        if ei.shape[1] < 4:
            continue

        src, dst = ei[0], ei[1]
        mask = (src < len(drug_imp)) & (dst < len(drug_imp)) & (src != dst)
        src, dst = src[mask], dst[mask]

        if len(src) < 10:
            continue

        edge_imp = drug_imp[src] + drug_imp[dst]

        # Deduplicate
        seen = set()
        unique_edges = []
        for idx in np.argsort(edge_imp)[::-1]:
            s, d = int(src[idx]), int(dst[idx])
            key = (min(s, d), max(s, d))
            if key not in seen:
                seen.add(key)
                unique_edges.append((s, d, edge_imp[idx]))

        if len(unique_edges) < 10:
            continue

        # Top-10 bonds
        top10 = unique_edges[:10]
        # Random-10 bonds
        rand_indices = np.random.choice(len(unique_edges), min(10, len(unique_edges)), replace=False)
        rand10 = [unique_edges[i] for i in rand_indices]

        # Check heteroatom involvement
        def hetero_fraction(edge_list):
            count = 0
            for s, d, _ in edge_list:
                s_sym = mol.GetAtomWithIdx(int(s)).GetSymbol() if int(s) < num_atoms else 'C'
                d_sym = mol.GetAtomWithIdx(int(d)).GetSymbol() if int(d) < num_atoms else 'C'
                if s_sym != 'C' or d_sym != 'C':
                    count += 1
            return count / len(edge_list) if edge_list else 0

        def aromatic_fraction(edge_list):
            count = 0
            for s, d, _ in edge_list:
                s_aro = mol.GetAtomWithIdx(int(s)).GetIsAromatic() if int(s) < num_atoms else False
                d_aro = mol.GetAtomWithIdx(int(d)).GetIsAromatic() if int(d) < num_atoms else False
                if s_aro or d_aro:
                    count += 1
            return count / len(edge_list) if edge_list else 0

        top10_hetero_fracs.append(hetero_fraction(top10))
        random10_hetero_fracs.append(hetero_fraction(rand10))
        top10_aromatic_fracs.append(aromatic_fraction(top10))
        random10_aromatic_fracs.append(aromatic_fraction(rand10))

    # Stats
    if top10_hetero_fracs:
        u_h, p_h = mannwhitneyu(top10_hetero_fracs, random10_hetero_fracs, alternative='greater')
        u_a, p_a = mannwhitneyu(top10_aromatic_fracs, random10_aromatic_fracs, alternative='two-sided')

        print(f"  Heteroatom involvement in top-10 bonds vs random-10:")
        print(f"    Top-10:    mean={np.mean(top10_hetero_fracs):.4f}")
        print(f"    Random-10: mean={np.mean(random10_hetero_fracs):.4f}")
        print(f"    Mann-Whitney p={p_h:.4e}")
        print()
        print(f"  Aromatic involvement in top-10 bonds vs random-10:")
        print(f"    Top-10:    mean={np.mean(top10_aromatic_fracs):.4f}")
        print(f"    Random-10: mean={np.mean(random10_aromatic_fracs):.4f}")
        print(f"    Mann-Whitney p={p_a:.4e}")
        print()

    return {
        'top10_hetero': top10_hetero_fracs,
        'rand10_hetero': random10_hetero_fracs,
        'top10_aromatic': top10_aromatic_fracs,
        'rand10_aromatic': random10_aromatic_fracs,
        'hetero_p': p_h if top10_hetero_fracs else None,
        'aromatic_p': p_a if top10_hetero_fracs else None,
    }


# =====================================================================
# TEST 3: DEGREE BIAS CHECK
# =====================================================================

def test_degree_bias(ll_results, drug_smiles_map, data_new, dataset, save_dir):
    """
    Check whether node importance is just a proxy for node degree.
    Low correlation = model learned genuine structure, not just topology.
    """
    print("=" * 70)
    print("TEST 3: DEGREE BIAS CHECK")
    print("=" * 70)
    print()

    drug_degree_corrs = []
    pro_degree_corrs = []

    for s_idx, result in enumerate(ll_results):
        drug_id = result['drug_id']
        protein_id = result['protein_id']
        drug_imp = result['drug_node_importance']
        pro_imp = result['protein_node_importance']
        if isinstance(drug_imp, torch.Tensor):
            drug_imp = drug_imp.numpy()
        if isinstance(pro_imp, torch.Tensor):
            pro_imp = pro_imp.numpy()

        # --- Drug degree ---
        smiles = drug_smiles_map.get(drug_id)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                num_atoms = mol.GetNumAtoms()
                degrees = np.array([mol.GetAtomWithIdx(i).GetDegree() for i in range(num_atoms)])
                imp_trimmed = drug_imp[:num_atoms]
                if len(degrees) > 3 and np.std(degrees) > 0 and np.std(imp_trimmed) > 0:
                    rho, _ = spearmanr(degrees, imp_trimmed)
                    drug_degree_corrs.append(rho)

        # --- Protein degree ---
        try:
            pro_data, _ = get_protein_graph_data(protein_id, data_new, dataset)
            ei = pro_data.edge_index.numpy()
            num_residues = pro_data.x.shape[0]
            degrees_p = np.zeros(num_residues)
            for i in range(ei.shape[1]):
                if ei[0, i] < num_residues:
                    degrees_p[ei[0, i]] += 1
            imp_p = pro_imp[:num_residues]
            if np.std(degrees_p) > 0 and np.std(imp_p) > 0:
                rho_p, _ = spearmanr(degrees_p, imp_p)
                pro_degree_corrs.append(rho_p)
        except Exception:
            pass

    print(f"  Drug importance-degree Spearman:    "
          f"mean={np.mean(drug_degree_corrs):.4f}, std={np.std(drug_degree_corrs):.4f}")
    print(f"  Protein importance-degree Spearman: "
          f"mean={np.mean(pro_degree_corrs):.4f}, std={np.std(pro_degree_corrs):.4f}")
    print()

    # Interpretation
    drug_mean = np.mean(drug_degree_corrs)
    pro_mean = np.mean(pro_degree_corrs)
    if abs(drug_mean) < 0.3:
        print(f"  Drug: LOW degree bias (rho={drug_mean:.3f}) -> importance is NOT driven by degree")
    elif abs(drug_mean) < 0.6:
        print(f"  Drug: MODERATE degree correlation (rho={drug_mean:.3f}) -> partial structural bias")
    else:
        print(f"  Drug: HIGH degree correlation (rho={drug_mean:.3f}) -> degree may dominate importance")

    if abs(pro_mean) < 0.3:
        print(f"  Protein: LOW degree bias (rho={pro_mean:.3f}) -> importance is NOT driven by degree")
    elif abs(pro_mean) < 0.6:
        print(f"  Protein: MODERATE degree correlation (rho={pro_mean:.3f}) -> partial structural bias")
    else:
        print(f"  Protein: HIGH degree correlation (rho={pro_mean:.3f}) -> degree may dominate importance")
    print()

    return {
        'drug_degree_corrs': drug_degree_corrs,
        'pro_degree_corrs': pro_degree_corrs,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    print()
    print("=" * 70)
    print("VALIDATION: EDGE IMPORTANCE SANITY CHECK (LL-6)")
    print("=" * 70)
    print()

    DATASET = "kiba"
    SAVE_DIR = os.path.join('results', 'validation', 'edge_sanity')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load results
    print("Loading saved results...")
    ll_results = torch.load(
        os.path.join('results', 'explainability_lowlevel', 'all_results.pt'),
        map_location='cpu'
    )
    print(f"  Loaded {len(ll_results)} samples")

    data_ckpt = torch.load('checkpoints/data_kiba.pt', map_location='cpu')
    data_new = data_ckpt['data_new']
    print(f"  Loaded data_new ({len(data_new)} entries)")

    # Build SMILES map
    drug_smiles_map = {}
    for item in data_new:
        if item[0] not in drug_smiles_map:
            drug_smiles_map[item[0]] = item[3]
    print()

    # --- Run all three tests ---
    consistency = test_consistency(ll_results, data_new, drug_smiles_map, DATASET, SAVE_DIR)
    bond_type = test_bond_type_discrimination(ll_results, drug_smiles_map, SAVE_DIR)
    degree_bias = test_degree_bias(ll_results, drug_smiles_map, data_new, DATASET, SAVE_DIR)

    # ===================================================================
    # VISUALIZATIONS
    # ===================================================================
    print("Generating visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # --- Plot 1: Consistency — Drug Pearson r distribution ---
    axes[0, 0].hist(consistency['drug_pearson_mean'], bins=20, color='steelblue', alpha=0.7,
                    edgecolor='black', label=f"mean={np.mean(consistency['drug_pearson_mean']):.4f}")
    axes[0, 0].axvline(x=np.mean(consistency['drug_pearson_mean']), color='red',
                        linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Pearson r (edge imp vs mean endpoint imp)')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].set_title('Drug: Edge-Node Consistency')
    axes[0, 0].legend()

    # --- Plot 2: Consistency — Protein Pearson r distribution ---
    axes[0, 1].hist(consistency['pro_pearson_mean'], bins=20, color='coral', alpha=0.7,
                    edgecolor='black', label=f"mean={np.mean(consistency['pro_pearson_mean']):.4f}")
    axes[0, 1].axvline(x=np.mean(consistency['pro_pearson_mean']), color='red',
                        linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Pearson r (edge imp vs mean endpoint imp)')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].set_title('Protein: Edge-Node Consistency')
    axes[0, 1].legend()

    # --- Plot 3: Bond-type discrimination ---
    if bond_type['top10_hetero']:
        bp = axes[0, 2].boxplot(
            [bond_type['top10_hetero'], bond_type['rand10_hetero'],
             bond_type['top10_aromatic'], bond_type['rand10_aromatic']],
            labels=['Top-10\nHetero', 'Rand-10\nHetero', 'Top-10\nAromatic', 'Rand-10\nAromatic'],
            patch_artist=True
        )
        colors_bp = ['steelblue', 'gray', 'coral', 'lightgray']
        for patch, c in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        axes[0, 2].set_ylabel('Fraction of Bonds')
        hetero_p = bond_type.get('hetero_p')
        p_str = f"p={hetero_p:.4e}" if hetero_p else ""
        axes[0, 2].set_title(f'Bond-Type Discrimination\n(Hetero: {p_str})')

    # --- Plot 4: Degree bias — Drug ---
    axes[1, 0].hist(degree_bias['drug_degree_corrs'], bins=20, color='steelblue', alpha=0.7,
                    edgecolor='black')
    axes[1, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=np.mean(degree_bias['drug_degree_corrs']), color='red',
                        linestyle='--', linewidth=2,
                        label=f"mean={np.mean(degree_bias['drug_degree_corrs']):.3f}")
    axes[1, 0].set_xlabel('Spearman rho (importance vs degree)')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_title('Drug: Degree Bias Check')
    axes[1, 0].legend()

    # --- Plot 5: Degree bias — Protein ---
    axes[1, 1].hist(degree_bias['pro_degree_corrs'], bins=20, color='coral', alpha=0.7,
                    edgecolor='black')
    axes[1, 1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(x=np.mean(degree_bias['pro_degree_corrs']), color='red',
                        linestyle='--', linewidth=2,
                        label=f"mean={np.mean(degree_bias['pro_degree_corrs']):.3f}")
    axes[1, 1].set_xlabel('Spearman rho (importance vs degree)')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('Protein: Degree Bias Check')
    axes[1, 1].legend()

    # --- Plot 6: Summary table ---
    axes[1, 2].axis('off')
    summary_data = [
        ['Metric', 'Drug', 'Protein'],
        ['Edge-Node r (mean ep.)',
         f"{np.mean(consistency['drug_pearson_mean']):.4f}",
         f"{np.mean(consistency['pro_pearson_mean']):.4f}"],
        ['Edge-Node r (max ep.)',
         f"{np.mean(consistency['drug_pearson_max']):.4f}",
         f"{np.mean(consistency['pro_pearson_max']):.4f}"],
        ['Degree-Imp Spearman',
         f"{np.mean(degree_bias['drug_degree_corrs']):.4f}",
         f"{np.mean(degree_bias['pro_degree_corrs']):.4f}"],
    ]
    if bond_type['top10_hetero']:
        summary_data.append([
            'Top-10 Hetero Frac',
            f"{np.mean(bond_type['top10_hetero']):.4f}",
            'N/A'
        ])
        summary_data.append([
            'Rand-10 Hetero Frac',
            f"{np.mean(bond_type['rand10_hetero']):.4f}",
            'N/A'
        ])

    table = axes[1, 2].table(
        cellText=summary_data[1:],
        colLabels=summary_data[0],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    axes[1, 2].set_title('Summary Table', fontsize=14, fontweight='bold')

    plt.suptitle('LL-6: Edge Importance Sanity Check', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, 'edge_sanity_check.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # ===================================================================
    # SAVE RESULTS & REPORT
    # ===================================================================
    torch.save({
        'consistency': consistency,
        'bond_type': bond_type,
        'degree_bias': degree_bias,
    }, os.path.join(SAVE_DIR, 'edge_sanity_results.pt'))

    lines = []
    lines.append("=" * 70)
    lines.append("LL-6: EDGE IMPORTANCE SANITY CHECK REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("TEST 1: EDGE-NODE CONSISTENCY (Pearson r)")
    lines.append(f"  Drug (mean endpoint):    r={np.mean(consistency['drug_pearson_mean']):.4f}")
    lines.append(f"  Drug (max endpoint):     r={np.mean(consistency['drug_pearson_max']):.4f}")
    lines.append(f"  Protein (mean endpoint): r={np.mean(consistency['pro_pearson_mean']):.4f}")
    lines.append(f"  Protein (max endpoint):  r={np.mean(consistency['pro_pearson_max']):.4f}")
    lines.append("")
    lines.append("TEST 2: BOND-TYPE DISCRIMINATION")
    if bond_type['top10_hetero']:
        lines.append(f"  Top-10 heteroatom frac:  {np.mean(bond_type['top10_hetero']):.4f}")
        lines.append(f"  Rand-10 heteroatom frac: {np.mean(bond_type['rand10_hetero']):.4f}")
        lines.append(f"  Mann-Whitney p={bond_type['hetero_p']:.4e}")
    lines.append("")
    lines.append("TEST 3: DEGREE BIAS")
    lines.append(f"  Drug importance-degree rho:    {np.mean(degree_bias['drug_degree_corrs']):.4f}")
    lines.append(f"  Protein importance-degree rho: {np.mean(degree_bias['pro_degree_corrs']):.4f}")
    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    report_path = os.path.join(SAVE_DIR, 'edge_sanity_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    print()
    print("=" * 70)
    print("EDGE IMPORTANCE SANITY CHECK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
