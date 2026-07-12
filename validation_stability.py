"""
=================================================================
VALIDATION: STABILITY & DISTRIBUTION TESTS (LL-3 & LL-5)
=================================================================

LL-3: Attribution Stability
  - Re-run IG 5 times on 10 samples to verify deterministic results.
  - Compute Spearman rank correlation and Jaccard similarity of top-15.

LL-5: Positive vs Negative Attribution Distribution
  - Compare attribution magnitudes between positive and negative pairs.
  - Mann-Whitney U test and effect size (Cohen's d).
=================================================================
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu
from itertools import combinations

from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool as gep

# Local imports
from NodeRepresentation import GNNNet
from model import H2GNN
import opt

from explainability_lowlevel import (
    build_drug_protein_maps,
    get_interaction_pairs,
    get_drug_graph_data,
    get_protein_graph_data,
    DrugGATWrapper,
    ProteinGATWrapper,
    compute_drug_attributions,
    compute_protein_attributions,
)


# =====================================================================
# LL-3: STABILITY TEST
# =====================================================================

def run_stability_test(gnnnet, data_new, dataset, drugmap, ll_results,
                       device, save_dir, n_samples=10, n_reruns=5, n_steps=200):
    """
    Re-run IG attributions multiple times on the same samples
    and check consistency.
    """
    print("=" * 70)
    print("LL-3: ATTRIBUTION STABILITY TEST")
    print("=" * 70)
    print(f"  Samples: {n_samples}, Re-runs: {n_reruns}, IG steps: {n_steps}")
    print()

    # Select the first n_samples from saved results
    test_samples = ll_results[:n_samples]

    drug_spearman_results = []
    drug_jaccard_results = []
    protein_spearman_results = []
    protein_jaccard_results = []

    for s_idx, result in enumerate(test_samples):
        drug_id = result['drug_id']
        protein_id = result['protein_id']
        print(f"  Sample {s_idx}: Drug={drug_id}, Protein={protein_id}")

        # --- Drug stability ---
        try:
            drug_data, smiles = get_drug_graph_data(drug_id, data_new, drugmap)
            drug_data = drug_data.to(device)

            drug_importances = []
            for run in range(n_reruns):
                wrapper = DrugGATWrapper(gnnnet, drug_data.edge_index).to(device)
                drug_features_input = drug_data.x.clone().requires_grad_(True)
                _, node_imp = compute_drug_attributions(wrapper, drug_features_input, n_steps=n_steps)
                drug_importances.append(node_imp.detach().cpu().numpy())

            # Pairwise Spearman correlations
            pairwise_rhos = []
            for (i, j) in combinations(range(n_reruns), 2):
                rho, _ = spearmanr(drug_importances[i], drug_importances[j])
                pairwise_rhos.append(rho)

            # Pairwise Jaccard of top-15
            pairwise_jaccard = []
            for (i, j) in combinations(range(n_reruns), 2):
                top_i = set(np.argsort(drug_importances[i])[::-1][:15])
                top_j = set(np.argsort(drug_importances[j])[::-1][:15])
                jaccard = len(top_i & top_j) / len(top_i | top_j)
                pairwise_jaccard.append(jaccard)

            mean_rho = np.mean(pairwise_rhos)
            mean_jaccard = np.mean(pairwise_jaccard)
            drug_spearman_results.append(mean_rho)
            drug_jaccard_results.append(mean_jaccard)
            print(f"    Drug:    Spearman rho={mean_rho:.4f}, Jaccard={mean_jaccard:.4f}")
        except Exception as e:
            print(f"    Drug stability failed: {e}")

        # --- Protein stability ---
        try:
            pro_data, sequence = get_protein_graph_data(protein_id, data_new, dataset)
            pro_data = pro_data.to(device)

            pro_importances = []
            for run in range(n_reruns):
                wrapper = ProteinGATWrapper(gnnnet, pro_data.edge_index).to(device)
                pro_features_input = pro_data.x.clone().requires_grad_(True)
                _, node_imp = compute_protein_attributions(wrapper, pro_features_input, n_steps=n_steps)
                pro_importances.append(node_imp.detach().cpu().numpy())

            pairwise_rhos = []
            for (i, j) in combinations(range(n_reruns), 2):
                rho, _ = spearmanr(pro_importances[i], pro_importances[j])
                pairwise_rhos.append(rho)

            pairwise_jaccard = []
            for (i, j) in combinations(range(n_reruns), 2):
                top_i = set(np.argsort(pro_importances[i])[::-1][:15])
                top_j = set(np.argsort(pro_importances[j])[::-1][:15])
                jaccard = len(top_i & top_j) / len(top_i | top_j)
                pairwise_jaccard.append(jaccard)

            mean_rho = np.mean(pairwise_rhos)
            mean_jaccard = np.mean(pairwise_jaccard)
            protein_spearman_results.append(mean_rho)
            protein_jaccard_results.append(mean_jaccard)
            print(f"    Protein: Spearman rho={mean_rho:.4f}, Jaccard={mean_jaccard:.4f}")
        except Exception as e:
            print(f"    Protein stability failed: {e}")

        print()

    # --- Summary ---
    print("-" * 50)
    print("STABILITY SUMMARY")
    print("-" * 50)
    if drug_spearman_results:
        print(f"  Drug Spearman rho:   mean={np.mean(drug_spearman_results):.4f}, "
              f"std={np.std(drug_spearman_results):.4f}, min={np.min(drug_spearman_results):.4f}")
        print(f"  Drug Jaccard top-15: mean={np.mean(drug_jaccard_results):.4f}, "
              f"std={np.std(drug_jaccard_results):.4f}, min={np.min(drug_jaccard_results):.4f}")
    if protein_spearman_results:
        print(f"  Prot Spearman rho:   mean={np.mean(protein_spearman_results):.4f}, "
              f"std={np.std(protein_spearman_results):.4f}, min={np.min(protein_spearman_results):.4f}")
        print(f"  Prot Jaccard top-15: mean={np.mean(protein_jaccard_results):.4f}, "
              f"std={np.std(protein_jaccard_results):.4f}, min={np.min(protein_jaccard_results):.4f}")
    print()

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Spearman box plot
    data_to_plot = []
    labels_to_plot = []
    if drug_spearman_results:
        data_to_plot.append(drug_spearman_results)
        labels_to_plot.append(f'Drug\n(n={len(drug_spearman_results)})')
    if protein_spearman_results:
        data_to_plot.append(protein_spearman_results)
        labels_to_plot.append(f'Protein\n(n={len(protein_spearman_results)})')

    bp1 = axes[0].boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
    colors = ['steelblue', 'coral']
    for patch, color in zip(bp1['boxes'], colors[:len(data_to_plot)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='rho=0.95 threshold')
    axes[0].set_ylabel('Spearman rho (pairwise)')
    axes[0].set_title('LL-3: Attribution Rank Stability\n(5 re-runs, pairwise Spearman rho)')
    axes[0].set_ylim(0.8, 1.02)
    axes[0].legend()

    # Jaccard box plot
    data_to_plot2 = []
    labels_to_plot2 = []
    if drug_jaccard_results:
        data_to_plot2.append(drug_jaccard_results)
        labels_to_plot2.append(f'Drug\n(n={len(drug_jaccard_results)})')
    if protein_jaccard_results:
        data_to_plot2.append(protein_jaccard_results)
        labels_to_plot2.append(f'Protein\n(n={len(protein_jaccard_results)})')

    bp2 = axes[1].boxplot(data_to_plot2, labels=labels_to_plot2, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors[:len(data_to_plot2)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Jaccard=0.8 threshold')
    axes[1].set_ylabel('Jaccard Similarity (top-15)')
    axes[1].set_title('LL-3: Top-15 Set Stability\n(5 re-runs, pairwise Jaccard)')
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'stability_test.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    return {
        'drug_spearman': drug_spearman_results,
        'drug_jaccard': drug_jaccard_results,
        'protein_spearman': protein_spearman_results,
        'protein_jaccard': protein_jaccard_results,
    }


# =====================================================================
# LL-5: POSITIVE VS NEGATIVE DISTRIBUTION
# =====================================================================

def run_distribution_test(ll_results, save_dir):
    """
    Compare attribution magnitude distributions between
    positive (interacting) and negative (non-interacting) pairs.
    """
    print("=" * 70)
    print("LL-5: POSITIVE vs NEGATIVE ATTRIBUTION DISTRIBUTION")
    print("=" * 70)
    print()

    pos_results = [r for r in ll_results if r['ground_truth'] == 1]
    neg_results = [r for r in ll_results if r['ground_truth'] == 0]
    print(f"  Positive samples: {len(pos_results)}")
    print(f"  Negative samples: {len(neg_results)}")
    print()

    # --- Drug Attribution Magnitude ---
    pos_drug_means = []
    neg_drug_means = []
    for r in pos_results:
        imp = r['drug_node_importance']
        if isinstance(imp, torch.Tensor):
            imp = imp.numpy()
        pos_drug_means.append(np.mean(imp))
    for r in neg_results:
        imp = r['drug_node_importance']
        if isinstance(imp, torch.Tensor):
            imp = imp.numpy()
        neg_drug_means.append(np.mean(imp))

    # --- Protein Attribution Magnitude ---
    pos_pro_means = []
    neg_pro_means = []
    for r in pos_results:
        imp = r['protein_node_importance']
        if isinstance(imp, torch.Tensor):
            imp = imp.numpy()
        pos_pro_means.append(np.mean(imp))
    for r in neg_results:
        imp = r['protein_node_importance']
        if isinstance(imp, torch.Tensor):
            imp = imp.numpy()
        neg_pro_means.append(np.mean(imp))

    # --- Drug Attribution Max (peak importance) ---
    pos_drug_max = []
    neg_drug_max = []
    for r in pos_results:
        imp = r['drug_node_importance']
        if isinstance(imp, torch.Tensor):
            imp = imp.numpy()
        pos_drug_max.append(np.max(imp))
    for r in neg_results:
        imp = r['drug_node_importance']
        if isinstance(imp, torch.Tensor):
            imp = imp.numpy()
        neg_drug_max.append(np.max(imp))

    # --- Protein Attribution Max ---
    pos_pro_max = []
    neg_pro_max = []
    for r in pos_results:
        imp = r['protein_node_importance']
        if isinstance(imp, torch.Tensor):
            imp = imp.numpy()
        pos_pro_max.append(np.max(imp))
    for r in neg_results:
        imp = r['protein_node_importance']
        if isinstance(imp, torch.Tensor):
            imp = imp.numpy()
        neg_pro_max.append(np.max(imp))

    # --- Attribution Entropy (how spread out the attribution is) ---
    def attribution_entropy(imp):
        """Normalized entropy of the importance distribution."""
        imp = np.abs(imp)
        total = np.sum(imp)
        if total == 0:
            return 0.0
        probs = imp / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(imp))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    pos_drug_entropy = []
    neg_drug_entropy = []
    pos_pro_entropy = []
    neg_pro_entropy = []
    for r in pos_results:
        imp = r['drug_node_importance']
        if isinstance(imp, torch.Tensor): imp = imp.numpy()
        pos_drug_entropy.append(attribution_entropy(imp))
        imp = r['protein_node_importance']
        if isinstance(imp, torch.Tensor): imp = imp.numpy()
        pos_pro_entropy.append(attribution_entropy(imp))
    for r in neg_results:
        imp = r['drug_node_importance']
        if isinstance(imp, torch.Tensor): imp = imp.numpy()
        neg_drug_entropy.append(attribution_entropy(imp))
        imp = r['protein_node_importance']
        if isinstance(imp, torch.Tensor): imp = imp.numpy()
        neg_pro_entropy.append(attribution_entropy(imp))

    # --- Statistical Tests ---
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1-1)*np.std(group1)**2 + (n2-1)*np.std(group2)**2) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)

    tests = [
        ('Drug Mean Attribution', pos_drug_means, neg_drug_means),
        ('Drug Max Attribution', pos_drug_max, neg_drug_max),
        ('Drug Attribution Entropy', pos_drug_entropy, neg_drug_entropy),
        ('Protein Mean Attribution', pos_pro_means, neg_pro_means),
        ('Protein Max Attribution', pos_pro_max, neg_pro_max),
        ('Protein Attribution Entropy', pos_pro_entropy, neg_pro_entropy),
    ]

    print("-" * 70)
    print(f"{'Metric':<30s}  {'Pos Mean':>10s}  {'Neg Mean':>10s}  {'U-stat':>10s}  {'p-value':>12s}  {'Cohen d':>8s}")
    print("-" * 70)

    test_results = {}
    for name, pos_vals, neg_vals in tests:
        u_stat, p_val = mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')
        d = cohens_d(pos_vals, neg_vals)
        print(f"{name:<30s}  {np.mean(pos_vals):10.6f}  {np.mean(neg_vals):10.6f}  "
              f"{u_stat:10.1f}  {p_val:12.4e}  {d:8.3f}")
        test_results[name] = {
            'pos_mean': np.mean(pos_vals),
            'neg_mean': np.mean(neg_vals),
            'u_stat': u_stat,
            'p_value': p_val,
            'cohens_d': d,
        }
    print("-" * 70)
    print()

    # --- Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Drug mean
    axes[0, 0].boxplot([pos_drug_means, neg_drug_means], labels=['Positive', 'Negative'])
    p_val = test_results['Drug Mean Attribution']['p_value']
    axes[0, 0].set_title(f'Drug Mean Attribution\np={p_val:.4e}')
    axes[0, 0].set_ylabel('Mean Importance')

    # Drug max
    axes[0, 1].boxplot([pos_drug_max, neg_drug_max], labels=['Positive', 'Negative'])
    p_val = test_results['Drug Max Attribution']['p_value']
    axes[0, 1].set_title(f'Drug Max Attribution\np={p_val:.4e}')

    # Drug entropy
    axes[0, 2].boxplot([pos_drug_entropy, neg_drug_entropy], labels=['Positive', 'Negative'])
    p_val = test_results['Drug Attribution Entropy']['p_value']
    axes[0, 2].set_title(f'Drug Attribution Entropy\np={p_val:.4e}')

    # Protein mean
    axes[1, 0].boxplot([pos_pro_means, neg_pro_means], labels=['Positive', 'Negative'])
    p_val = test_results['Protein Mean Attribution']['p_value']
    axes[1, 0].set_title(f'Protein Mean Attribution\np={p_val:.4e}')
    axes[1, 0].set_ylabel('Mean Importance')

    # Protein max
    axes[1, 1].boxplot([pos_pro_max, neg_pro_max], labels=['Positive', 'Negative'])
    p_val = test_results['Protein Max Attribution']['p_value']
    axes[1, 1].set_title(f'Protein Max Attribution\np={p_val:.4e}')

    # Protein entropy
    axes[1, 2].boxplot([pos_pro_entropy, neg_pro_entropy], labels=['Positive', 'Negative'])
    p_val = test_results['Protein Attribution Entropy']['p_value']
    axes[1, 2].set_title(f'Protein Attribution Entropy\np={p_val:.4e}')

    plt.suptitle('LL-5: Attribution Distribution — Positive vs Negative Interactions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'pos_vs_neg_distributions.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    return test_results


# =====================================================================
# MAIN
# =====================================================================

def main():
    print()
    print("=" * 70)
    print("VALIDATION: STABILITY & DISTRIBUTION TESTS (LL-3 & LL-5)")
    print("=" * 70)
    print()

    DATASET = "kiba"
    SAVE_DIR = os.path.join('results', 'validation', 'stability')
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device('cpu')

    # --- Load saved results ---
    print("Loading saved explainability results...")
    ll_results = torch.load(
        os.path.join('results', 'explainability_lowlevel', 'all_results.pt'),
        map_location='cpu'
    )
    print(f"  Loaded {len(ll_results)} low-level results")

    # --- Load model for stability test ---
    print("Loading checkpoints...")
    data_ckpt = torch.load(f'checkpoints/data_{DATASET}.pt', map_location=device)
    data_new = data_ckpt['data_new']

    args = opt.parser.parse_args([])
    gnnnet = GNNNet().to(device)
    gnnnet.load_state_dict(torch.load(f'checkpoints/gnnnet_{DATASET}.pt', map_location=device))
    gnnnet.eval()
    print(f"  Loaded GNNNet")

    drugmap, proteinmap, _, _ = build_drug_protein_maps(data_new)
    print()

    # --- LL-3: Stability Test ---
    stability_results = run_stability_test(
        gnnnet, data_new, DATASET, drugmap, ll_results,
        device, SAVE_DIR, n_samples=10, n_reruns=5, n_steps=200
    )

    # --- LL-5: Positive vs Negative Distribution ---
    distribution_results = run_distribution_test(ll_results, SAVE_DIR)

    # --- Save all results ---
    torch.save({
        'stability': stability_results,
        'distribution': distribution_results,
    }, os.path.join(SAVE_DIR, 'stability_distribution_results.pt'))

    # --- Write summary report ---
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("STABILITY & DISTRIBUTION VALIDATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("LL-3: ATTRIBUTION STABILITY")
    report_lines.append("-" * 40)
    if stability_results['drug_spearman']:
        report_lines.append(f"  Drug Spearman rho:  mean={np.mean(stability_results['drug_spearman']):.4f}")
        report_lines.append(f"  Drug Jaccard:       mean={np.mean(stability_results['drug_jaccard']):.4f}")
    if stability_results['protein_spearman']:
        report_lines.append(f"  Prot Spearman rho:  mean={np.mean(stability_results['protein_spearman']):.4f}")
        report_lines.append(f"  Prot Jaccard:       mean={np.mean(stability_results['protein_jaccard']):.4f}")
    report_lines.append("")
    report_lines.append("LL-5: POSITIVE vs NEGATIVE DISTRIBUTION")
    report_lines.append("-" * 40)
    for name, vals in distribution_results.items():
        sig = '***' if vals['p_value'] < 0.001 else '**' if vals['p_value'] < 0.01 else '*' if vals['p_value'] < 0.05 else 'n.s.'
        report_lines.append(f"  {name}: p={vals['p_value']:.4e}, d={vals['cohens_d']:.3f} [{sig}]")
    report_lines.append("")
    report_lines.append("=" * 70)

    report = "\n".join(report_lines)
    report_path = os.path.join(SAVE_DIR, 'stability_distribution_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n  Report saved: {report_path}")

    print()
    print("=" * 70)
    print("STABILITY & DISTRIBUTION TESTS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
