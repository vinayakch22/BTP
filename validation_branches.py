"""
=================================================================
VALIDATION: BRANCH ANALYSIS & FEATURE CONSISTENCY (HL-5 & HL-6)
=================================================================

HL-5: AE vs IGAE Branch Contribution Analysis
  - Compare total absolute attribution between AE (L4) and IGAE (L5).
  - Pearson correlation between AE and IGAE per-feature attributions.
  - Correlation between branch dominance and alpha (from L1).

HL-6: Feature Consistency Across Samples
  - Count how often each feature dimension appears in top-20.
  - Jaccard similarity of top-20 features between same-drug/protein pairs.

Uses only saved results -- no model re-runs needed.
=================================================================
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict


# =====================================================================
# HL-5: AE vs IGAE BRANCH ANALYSIS
# =====================================================================

def validate_branch_analysis(hl_results, save_dir):
    """Compare AE and IGAE branch contributions."""
    print("=" * 70)
    print("HL-5: AE vs IGAE BRANCH CONTRIBUTION ANALYSIS")
    print("=" * 70)
    print()

    n = len(hl_results)

    # Total absolute attribution per branch per sample
    ae_drug_totals = []
    ae_pro_totals = []
    igae_drug_totals = []
    igae_pro_totals = []
    alpha_drugs = []
    alpha_prots = []
    ground_truths = []

    for r in hl_results:
        ae_drug_totals.append(np.sum(np.abs(r['level4']['attr_features_drug'])))
        ae_pro_totals.append(np.sum(np.abs(r['level4']['attr_features_protein'])))
        igae_drug_totals.append(np.sum(np.abs(r['level5']['attr_features_drug'])))
        igae_pro_totals.append(np.sum(np.abs(r['level5']['attr_features_protein'])))
        alpha_drugs.append(r['level1']['alpha_mean_drug'])
        alpha_prots.append(r['level1']['alpha_mean_protein'])
        ground_truths.append(r['ground_truth'])

    ae_drug_totals = np.array(ae_drug_totals)
    ae_pro_totals = np.array(ae_pro_totals)
    igae_drug_totals = np.array(igae_drug_totals)
    igae_pro_totals = np.array(igae_pro_totals)

    # --- Branch dominance ratio ---
    drug_ratio = igae_drug_totals / (ae_drug_totals + 1e-10)
    pro_ratio = igae_pro_totals / (ae_pro_totals + 1e-10)

    print(f"  Drug AE total:   mean={np.mean(ae_drug_totals):.6f}, std={np.std(ae_drug_totals):.6f}")
    print(f"  Drug IGAE total: mean={np.mean(igae_drug_totals):.6f}, std={np.std(igae_drug_totals):.6f}")
    print(f"  Drug IGAE/AE:    mean={np.mean(drug_ratio):.2f}x")
    print()
    print(f"  Prot AE total:   mean={np.mean(ae_pro_totals):.6f}, std={np.std(ae_pro_totals):.6f}")
    print(f"  Prot IGAE total: mean={np.mean(igae_pro_totals):.6f}, std={np.std(igae_pro_totals):.6f}")
    print(f"  Prot IGAE/AE:    mean={np.mean(pro_ratio):.2f}x")
    print()

    # --- Per-feature correlation between AE and IGAE ---
    feature_correlations_drug = []
    feature_correlations_pro = []
    for r in hl_results:
        ae_d = r['level4']['attr_features_drug']
        igae_d = r['level5']['attr_features_drug']
        if np.std(ae_d) > 0 and np.std(igae_d) > 0:
            corr, _ = pearsonr(ae_d, igae_d)
            feature_correlations_drug.append(corr)

        ae_p = r['level4']['attr_features_protein']
        igae_p = r['level5']['attr_features_protein']
        if np.std(ae_p) > 0 and np.std(igae_p) > 0:
            corr, _ = pearsonr(ae_p, igae_p)
            feature_correlations_pro.append(corr)

    print(f"  AE-IGAE per-feature Pearson (Drug):    mean={np.mean(feature_correlations_drug):.4f}")
    print(f"  AE-IGAE per-feature Pearson (Protein): mean={np.mean(feature_correlations_pro):.4f}")
    print()

    # --- Correlation between alpha and IGAE/AE ratio ---
    rho_drug_alpha, p_drug_alpha = spearmanr(alpha_drugs, drug_ratio)
    rho_pro_alpha, p_pro_alpha = spearmanr(alpha_prots, pro_ratio)
    print(f"  Alpha vs IGAE/AE ratio correlation:")
    print(f"    Drug:    Spearman rho={rho_drug_alpha:.4f}, p={p_drug_alpha:.4e}")
    print(f"    Protein: Spearman rho={rho_pro_alpha:.4f}, p={p_pro_alpha:.4e}")
    print()

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['green' if gt == 1 else 'red' for gt in ground_truths]

    # AE vs IGAE scatter (Drug)
    axes[0, 0].scatter(ae_drug_totals, igae_drug_totals, c=colors, alpha=0.6, s=40)
    max_val = max(np.max(ae_drug_totals), np.max(igae_drug_totals)) * 1.1
    axes[0, 0].plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    axes[0, 0].set_xlabel('AE Total Attribution (Drug)')
    axes[0, 0].set_ylabel('IGAE Total Attribution (Drug)')
    axes[0, 0].set_title(f'Drug: AE vs IGAE Attribution\nIGAE/AE ratio={np.mean(drug_ratio):.2f}x')
    axes[0, 0].legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Positive'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Negative')
    ])

    # AE vs IGAE scatter (Protein)
    axes[0, 1].scatter(ae_pro_totals, igae_pro_totals, c=colors, alpha=0.6, s=40)
    max_val = max(np.max(ae_pro_totals), np.max(igae_pro_totals)) * 1.1
    axes[0, 1].plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    axes[0, 1].set_xlabel('AE Total Attribution (Protein)')
    axes[0, 1].set_ylabel('IGAE Total Attribution (Protein)')
    axes[0, 1].set_title(f'Protein: AE vs IGAE Attribution\nIGAE/AE ratio={np.mean(pro_ratio):.2f}x')

    # Feature correlation histogram
    axes[1, 0].hist(feature_correlations_drug, bins=20, alpha=0.6, label='Drug', color='steelblue')
    axes[1, 0].hist(feature_correlations_pro, bins=20, alpha=0.6, label='Protein', color='coral')
    axes[1, 0].set_xlabel('Pearson Correlation (AE vs IGAE per feature)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Per-Feature AE-IGAE Correlation\n(low = complementary, high = redundant)')
    axes[1, 0].legend()
    axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.3)

    # Alpha vs IGAE/AE ratio
    axes[1, 1].scatter(alpha_drugs, drug_ratio, c=colors, alpha=0.6, s=40)
    z = np.polyfit(alpha_drugs, drug_ratio, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(min(alpha_drugs), max(alpha_drugs), 50)
    axes[1, 1].plot(x_line, p_line(x_line), 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('Drug Alpha (higher = more AE)')
    axes[1, 1].set_ylabel('IGAE/AE Attribution Ratio')
    axes[1, 1].set_title(f'Alpha vs Branch Dominance\nSpearman rho={rho_drug_alpha:.4f}, p={p_drug_alpha:.4e}')

    plt.suptitle('HL-5: AE vs IGAE Branch Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'branch_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    return {
        'drug_ae_mean': float(np.mean(ae_drug_totals)),
        'drug_igae_mean': float(np.mean(igae_drug_totals)),
        'drug_ratio_mean': float(np.mean(drug_ratio)),
        'pro_ratio_mean': float(np.mean(pro_ratio)),
        'feature_corr_drug_mean': float(np.mean(feature_correlations_drug)),
        'feature_corr_pro_mean': float(np.mean(feature_correlations_pro)),
        'alpha_ratio_rho_drug': rho_drug_alpha,
        'alpha_ratio_p_drug': p_drug_alpha,
    }


# =====================================================================
# HL-6: FEATURE CONSISTENCY
# =====================================================================

def validate_feature_consistency(hl_results, save_dir):
    """Check if the same features are consistently important across samples."""
    print("=" * 70)
    print("HL-6: FEATURE CONSISTENCY ACROSS SAMPLES")
    print("=" * 70)
    print()

    n = len(hl_results)

    # --- Frequency analysis: how often each feature appears in top-20 ---
    ae_drug_freq = defaultdict(int)
    ae_pro_freq = defaultdict(int)
    igae_drug_freq = defaultdict(int)
    igae_pro_freq = defaultdict(int)

    for r in hl_results:
        for dim in r['level4']['drug_top_features'][:20]:
            ae_drug_freq[dim] += 1
        for dim in r['level4']['protein_top_features'][:20]:
            ae_pro_freq[dim] += 1
        for dim in r['level5']['drug_top_features'][:20]:
            igae_drug_freq[dim] += 1
        for dim in r['level5']['protein_top_features'][:20]:
            igae_pro_freq[dim] += 1

    # Top-10 most frequently important features
    print("  AE Drug Top-10 Most Frequent Features (in top-20 across all samples):")
    sorted_ae_drug = sorted(ae_drug_freq.items(), key=lambda x: x[1], reverse=True)
    for rank, (dim, count) in enumerate(sorted_ae_drug[:10]):
        print(f"    {rank+1:2d}. Feature dim {dim:3d}: {count}/{n} samples ({100*count/n:.0f}%)")

    print()
    print("  IGAE Drug Top-10 Most Frequent Features:")
    sorted_igae_drug = sorted(igae_drug_freq.items(), key=lambda x: x[1], reverse=True)
    for rank, (dim, count) in enumerate(sorted_igae_drug[:10]):
        print(f"    {rank+1:2d}. Feature dim {dim:3d}: {count}/{n} samples ({100*count/n:.0f}%)")

    print()
    print("  AE Protein Top-10 Most Frequent Features:")
    sorted_ae_pro = sorted(ae_pro_freq.items(), key=lambda x: x[1], reverse=True)
    for rank, (dim, count) in enumerate(sorted_ae_pro[:10]):
        print(f"    {rank+1:2d}. Feature dim {dim:3d}: {count}/{n} samples ({100*count/n:.0f}%)")

    print()
    print("  IGAE Protein Top-10 Most Frequent Features:")
    sorted_igae_pro = sorted(igae_pro_freq.items(), key=lambda x: x[1], reverse=True)
    for rank, (dim, count) in enumerate(sorted_igae_pro[:10]):
        print(f"    {rank+1:2d}. Feature dim {dim:3d}: {count}/{n} samples ({100*count/n:.0f}%)")
    print()

    # --- Jaccard similarity between samples with the same drug ---
    drug_groups = defaultdict(list)
    protein_groups = defaultdict(list)
    for i, r in enumerate(hl_results):
        drug_groups[r['drug_id']].append(i)
        protein_groups[r['protein_id']].append(i)

    # Same-drug Jaccard (AE features)
    same_drug_jaccards = []
    for drug_id, indices in drug_groups.items():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                set_i = set(hl_results[indices[i]]['level4']['drug_top_features'][:20])
                set_j = set(hl_results[indices[j]]['level4']['drug_top_features'][:20])
                jaccard = len(set_i & set_j) / len(set_i | set_j)
                same_drug_jaccards.append(jaccard)

    # Different-drug Jaccard (random baseline)
    diff_drug_jaccards = []
    all_indices = list(range(n))
    np.random.seed(42)
    for _ in range(min(200, n*(n-1)//2)):
        i, j = np.random.choice(n, 2, replace=False)
        if hl_results[i]['drug_id'] != hl_results[j]['drug_id']:
            set_i = set(hl_results[i]['level4']['drug_top_features'][:20])
            set_j = set(hl_results[j]['level4']['drug_top_features'][:20])
            jaccard = len(set_i & set_j) / len(set_i | set_j)
            diff_drug_jaccards.append(jaccard)

    # Same-protein Jaccard
    same_pro_jaccards = []
    for protein_id, indices in protein_groups.items():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                set_i = set(hl_results[indices[i]]['level4']['protein_top_features'][:20])
                set_j = set(hl_results[indices[j]]['level4']['protein_top_features'][:20])
                jaccard = len(set_i & set_j) / len(set_i | set_j)
                same_pro_jaccards.append(jaccard)

    print(f"  Same-Drug Jaccard (AE top-20):     mean={np.mean(same_drug_jaccards):.4f} (n={len(same_drug_jaccards)} pairs)" if same_drug_jaccards else "  Same-Drug: No repeated drugs")
    print(f"  Diff-Drug Jaccard (AE top-20):     mean={np.mean(diff_drug_jaccards):.4f} (n={len(diff_drug_jaccards)} pairs)" if diff_drug_jaccards else "  Diff-Drug: N/A")
    print(f"  Same-Protein Jaccard (AE top-20):  mean={np.mean(same_pro_jaccards):.4f} (n={len(same_pro_jaccards)} pairs)" if same_pro_jaccards else "  Same-Protein: No repeated proteins")
    print()

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Feature frequency histogram (AE Drug)
    all_ae_drug_counts = [ae_drug_freq.get(d, 0) for d in range(160)]
    axes[0, 0].bar(range(160), all_ae_drug_counts, color='steelblue', alpha=0.7, width=1.0)
    axes[0, 0].set_xlabel('Feature Dimension Index')
    axes[0, 0].set_ylabel(f'Frequency (out of {n} samples)')
    axes[0, 0].set_title('AE Drug: Feature Frequency in Top-20')

    # Feature frequency histogram (IGAE Drug)
    all_igae_drug_counts = [igae_drug_freq.get(d, 0) for d in range(160)]
    axes[0, 1].bar(range(160), all_igae_drug_counts, color='coral', alpha=0.7, width=1.0)
    axes[0, 1].set_xlabel('Feature Dimension Index')
    axes[0, 1].set_ylabel(f'Frequency (out of {n} samples)')
    axes[0, 1].set_title('IGAE Drug: Feature Frequency in Top-20')

    # Same vs Different drug Jaccard
    if same_drug_jaccards and diff_drug_jaccards:
        bp = axes[1, 0].boxplot([same_drug_jaccards, diff_drug_jaccards],
                                 labels=['Same Drug', 'Diff Drug'], patch_artist=True)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('gray')
        for box in bp['boxes']:
            box.set_alpha(0.7)
        if len(same_drug_jaccards) >= 5 and len(diff_drug_jaccards) >= 5:
            u, p = stats.mannwhitneyu(same_drug_jaccards, diff_drug_jaccards, alternative='greater')
            axes[1, 0].set_title(f'Same-Drug vs Diff-Drug Consistency\nMann-Whitney p={p:.4e}')
        else:
            axes[1, 0].set_title('Same-Drug vs Diff-Drug Consistency')
        axes[1, 0].set_ylabel('Jaccard Similarity (AE top-20)')
    else:
        axes[1, 0].text(0.5, 0.5, 'Insufficient repeated drugs', ha='center', va='center')
        axes[1, 0].set_title('Same-Drug vs Diff-Drug Consistency')

    # Same vs Different protein Jaccard
    diff_pro_jaccards = []
    for _ in range(min(200, n*(n-1)//2)):
        i, j = np.random.choice(n, 2, replace=False)
        if hl_results[i]['protein_id'] != hl_results[j]['protein_id']:
            set_i = set(hl_results[i]['level4']['protein_top_features'][:20])
            set_j = set(hl_results[j]['level4']['protein_top_features'][:20])
            jaccard = len(set_i & set_j) / len(set_i | set_j)
            diff_pro_jaccards.append(jaccard)

    if same_pro_jaccards and diff_pro_jaccards:
        bp2 = axes[1, 1].boxplot([same_pro_jaccards, diff_pro_jaccards],
                                  labels=['Same Protein', 'Diff Protein'], patch_artist=True)
        bp2['boxes'][0].set_facecolor('coral')
        bp2['boxes'][1].set_facecolor('gray')
        for box in bp2['boxes']:
            box.set_alpha(0.7)
        if len(same_pro_jaccards) >= 5 and len(diff_pro_jaccards) >= 5:
            u, p = stats.mannwhitneyu(same_pro_jaccards, diff_pro_jaccards, alternative='greater')
            axes[1, 1].set_title(f'Same-Protein vs Diff-Protein Consistency\nMann-Whitney p={p:.4e}')
        else:
            axes[1, 1].set_title('Same-Protein vs Diff-Protein Consistency')
        axes[1, 1].set_ylabel('Jaccard Similarity (AE top-20)')
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient repeated proteins', ha='center', va='center')
        axes[1, 1].set_title('Same-Protein vs Diff-Protein Consistency')

    plt.suptitle('HL-6: Feature Consistency Across Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'feature_consistency.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    return {
        'ae_drug_top_features': [d for d, _ in sorted_ae_drug[:10]],
        'igae_drug_top_features': [d for d, _ in sorted_igae_drug[:10]],
        'same_drug_jaccard': float(np.mean(same_drug_jaccards)) if same_drug_jaccards else None,
        'diff_drug_jaccard': float(np.mean(diff_drug_jaccards)) if diff_drug_jaccards else None,
        'same_pro_jaccard': float(np.mean(same_pro_jaccards)) if same_pro_jaccards else None,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    print()
    print("=" * 70)
    print("VALIDATION: BRANCH ANALYSIS & FEATURE CONSISTENCY (HL-5, HL-6)")
    print("=" * 70)
    print()

    SAVE_DIR = os.path.join('results', 'validation', 'branches')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load saved results
    print("Loading saved prediction-level results...")
    hl_results = torch.load(
        os.path.join('results', 'explainability_highlevel', 'all_results.pt'),
        map_location='cpu'
    )
    print(f"  Loaded {len(hl_results)} high-level results")
    print()

    # --- HL-5 ---
    branch_results = validate_branch_analysis(hl_results, SAVE_DIR)
    print()

    # --- HL-6 ---
    consistency_results = validate_feature_consistency(hl_results, SAVE_DIR)
    print()

    # --- Save ---
    torch.save({
        'branches': branch_results,
        'consistency': consistency_results,
    }, os.path.join(SAVE_DIR, 'branches_validation_results.pt'))

    # --- Report ---
    lines = []
    lines.append("=" * 70)
    lines.append("BRANCH ANALYSIS & FEATURE CONSISTENCY REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("HL-5: AE vs IGAE BRANCHES")
    lines.append(f"  Drug IGAE/AE ratio:             {branch_results['drug_ratio_mean']:.2f}x")
    lines.append(f"  Prot IGAE/AE ratio:             {branch_results['pro_ratio_mean']:.2f}x")
    lines.append(f"  AE-IGAE feature corr (Drug):    {branch_results['feature_corr_drug_mean']:.4f}")
    lines.append(f"  AE-IGAE feature corr (Protein): {branch_results['feature_corr_pro_mean']:.4f}")
    lines.append(f"  Alpha vs ratio Spearman (Drug):  rho={branch_results['alpha_ratio_rho_drug']:.4f}")
    lines.append("")
    lines.append("HL-6: FEATURE CONSISTENCY")
    if consistency_results['same_drug_jaccard'] is not None:
        lines.append(f"  Same-Drug Jaccard:              {consistency_results['same_drug_jaccard']:.4f}")
    if consistency_results['diff_drug_jaccard'] is not None:
        lines.append(f"  Diff-Drug Jaccard:              {consistency_results['diff_drug_jaccard']:.4f}")
    if consistency_results['same_pro_jaccard'] is not None:
        lines.append(f"  Same-Protein Jaccard:           {consistency_results['same_pro_jaccard']:.4f}")
    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    report_path = os.path.join(SAVE_DIR, 'branches_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    print()
    print("=" * 70)
    print("BRANCH ANALYSIS & FEATURE CONSISTENCY COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
