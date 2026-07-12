"""
=================================================================
VALIDATION: ATTENTION ANALYSIS (HL-1 & HL-2)
=================================================================

HL-1: Dynamic Fusion Gate (Alpha) Distribution Analysis
  - Compare alpha between drugs vs proteins (Mann-Whitney U)
  - Compare alpha between positive vs negative pairs
  - Correlation between alpha and prediction score

HL-2: Self-Attention Discriminative Power
  - AUC of self-attention score as standalone predictor
  - Mann-Whitney U between positive vs negative attention scores

Uses only saved results — no model re-runs needed.
=================================================================
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve


# =====================================================================
# HL-1: ALPHA DISTRIBUTION ANALYSIS
# =====================================================================

def validate_alpha_distribution(hl_results, save_dir):
    """Analyse the dynamic fusion gate (alpha) distributions."""
    print("=" * 70)
    print("HL-1: DYNAMIC FUSION GATE (ALPHA) DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print()

    alpha_drugs = [r['level1']['alpha_mean_drug'] for r in hl_results]
    alpha_prots = [r['level1']['alpha_mean_protein'] for r in hl_results]
    predictions = [r['prediction'] for r in hl_results]
    ground_truths = [r['ground_truth'] for r in hl_results]

    pos_results = [r for r in hl_results if r['ground_truth'] == 1]
    neg_results = [r for r in hl_results if r['ground_truth'] == 0]

    # --- Test 1: Drug vs Protein alpha ---
    u_dp, p_dp = mannwhitneyu(alpha_drugs, alpha_prots, alternative='two-sided')
    print(f"  Test 1: Drug alpha vs Protein alpha")
    print(f"    Drug alpha:    mean={np.mean(alpha_drugs):.4f}, std={np.std(alpha_drugs):.4f}")
    print(f"    Protein alpha: mean={np.mean(alpha_prots):.4f}, std={np.std(alpha_prots):.4f}")
    print(f"    Mann-Whitney U={u_dp:.1f}, p={p_dp:.4e}")
    print()

    # --- Test 2: Positive vs Negative alpha (drugs) ---
    pos_alpha_drugs = [r['level1']['alpha_mean_drug'] for r in pos_results]
    neg_alpha_drugs = [r['level1']['alpha_mean_drug'] for r in neg_results]
    if pos_alpha_drugs and neg_alpha_drugs:
        u_pn, p_pn = mannwhitneyu(pos_alpha_drugs, neg_alpha_drugs, alternative='two-sided')
        print(f"  Test 2: Positive vs Negative Drug alpha")
        print(f"    Positive: mean={np.mean(pos_alpha_drugs):.4f}")
        print(f"    Negative: mean={np.mean(neg_alpha_drugs):.4f}")
        print(f"    Mann-Whitney U={u_pn:.1f}, p={p_pn:.4e}")
        print()

    # --- Test 3: Positive vs Negative alpha (proteins) ---
    pos_alpha_prots = [r['level1']['alpha_mean_protein'] for r in pos_results]
    neg_alpha_prots = [r['level1']['alpha_mean_protein'] for r in neg_results]
    if pos_alpha_prots and neg_alpha_prots:
        u_pn2, p_pn2 = mannwhitneyu(pos_alpha_prots, neg_alpha_prots, alternative='two-sided')
        print(f"  Test 3: Positive vs Negative Protein alpha")
        print(f"    Positive: mean={np.mean(pos_alpha_prots):.4f}")
        print(f"    Negative: mean={np.mean(neg_alpha_prots):.4f}")
        print(f"    Mann-Whitney U={u_pn2:.1f}, p={p_pn2:.4e}")
        print()

    # --- Test 4: Alpha vs Prediction correlation ---
    rho_drug, p_rho_drug = spearmanr(alpha_drugs, predictions)
    rho_prot, p_rho_prot = spearmanr(alpha_prots, predictions)
    print(f"  Test 4: Alpha vs Prediction Score Correlation")
    print(f"    Drug alpha <-> Prediction:    rho={rho_drug:.4f}, p={p_rho_drug:.4e}")
    print(f"    Protein alpha <-> Prediction: rho={rho_prot:.4f}, p={p_rho_prot:.4e}")
    print()

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Violin: Drug vs Protein alpha
    parts = axes[0, 0].violinplot([alpha_drugs, alpha_prots], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['steelblue', 'coral'][i])
        pc.set_alpha(0.7)
    axes[0, 0].set_xticks([1, 2])
    axes[0, 0].set_xticklabels(['Drug α', 'Protein α'])
    axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_ylabel('Mean Alpha')
    axes[0, 0].set_title(f'Drug vs Protein Alpha\nMann-Whitney p={p_dp:.4e}')

    # Violin: Positive vs Negative (Drug alpha)
    if pos_alpha_drugs and neg_alpha_drugs:
        parts2 = axes[0, 1].violinplot([pos_alpha_drugs, neg_alpha_drugs], showmeans=True, showmedians=True)
        for i, pc in enumerate(parts2['bodies']):
            pc.set_facecolor(['green', 'red'][i])
            pc.set_alpha(0.6)
        axes[0, 1].set_xticks([1, 2])
        axes[0, 1].set_xticklabels(['Positive (GT=1)', 'Negative (GT=0)'])
        axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].set_ylabel('Drug Mean Alpha')
        axes[0, 1].set_title(f'Drug Alpha by Interaction Label\np={p_pn:.4e}')

    # Scatter: Drug alpha vs Prediction
    colors = ['green' if gt == 1 else 'red' for gt in ground_truths]
    axes[1, 0].scatter(alpha_drugs, predictions, c=colors, alpha=0.6, s=40)
    z = np.polyfit(alpha_drugs, predictions, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(min(alpha_drugs), max(alpha_drugs), 50)
    axes[1, 0].plot(x_line, p_line(x_line), 'k--', alpha=0.5)
    axes[1, 0].set_xlabel('Drug Mean Alpha')
    axes[1, 0].set_ylabel('Prediction Score')
    axes[1, 0].set_title(f'Drug Alpha vs Prediction\nSpearman rho={rho_drug:.4f}, p={p_rho_drug:.4e}')
    axes[1, 0].legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Positive'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Negative')
    ])

    # Scatter: Protein alpha vs Prediction
    axes[1, 1].scatter(alpha_prots, predictions, c=colors, alpha=0.6, s=40)
    z2 = np.polyfit(alpha_prots, predictions, 1)
    p_line2 = np.poly1d(z2)
    x_line2 = np.linspace(min(alpha_prots), max(alpha_prots), 50)
    axes[1, 1].plot(x_line2, p_line2(x_line2), 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('Protein Mean Alpha')
    axes[1, 1].set_ylabel('Prediction Score')
    axes[1, 1].set_title(f'Protein Alpha vs Prediction\nSpearman rho={rho_prot:.4f}, p={p_rho_prot:.4e}')
    axes[1, 1].legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Positive'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Negative')
    ])

    plt.suptitle('HL-1: Dynamic Fusion Gate (Alpha) Validation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'alpha_distribution_validation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    return {
        'drug_vs_protein_p': p_dp,
        'pos_vs_neg_drug_p': p_pn if pos_alpha_drugs else None,
        'pos_vs_neg_prot_p': p_pn2 if pos_alpha_prots else None,
        'drug_alpha_pred_rho': rho_drug,
        'drug_alpha_pred_p': p_rho_drug,
        'prot_alpha_pred_rho': rho_prot,
        'prot_alpha_pred_p': p_rho_prot,
    }


# =====================================================================
# HL-2: SELF-ATTENTION DISCRIMINATIVE POWER
# =====================================================================

def validate_self_attention(hl_results, save_dir):
    """Test if self-attention drug->protein score discriminates interactions."""
    print("=" * 70)
    print("HL-2: SELF-ATTENTION DISCRIMINATIVE POWER")
    print("=" * 70)
    print()

    d2p_scores = [r['level1']['self_attn_drug_to_protein'] for r in hl_results]
    p2d_scores = [r['level1']['self_attn_protein_to_drug'] for r in hl_results]
    ground_truths = [r['ground_truth'] for r in hl_results]

    # --- AUC as standalone predictor ---
    auc_d2p = roc_auc_score(ground_truths, d2p_scores)
    auc_p2d = roc_auc_score(ground_truths, p2d_scores)
    print(f"  Self-Attention as Standalone Predictor:")
    print(f"    Drug->Protein AUC: {auc_d2p:.4f}")
    print(f"    Protein->Drug AUC: {auc_p2d:.4f}")
    print()

    # --- Mann-Whitney U ---
    pos_d2p = [d2p_scores[i] for i in range(len(ground_truths)) if ground_truths[i] == 1]
    neg_d2p = [d2p_scores[i] for i in range(len(ground_truths)) if ground_truths[i] == 0]

    u_stat, p_val = mannwhitneyu(pos_d2p, neg_d2p, alternative='two-sided')
    print(f"  Drug->Protein Attention: Positive vs Negative")
    print(f"    Positive: mean={np.mean(pos_d2p):.6f}, std={np.std(pos_d2p):.6f}")
    print(f"    Negative: mean={np.mean(neg_d2p):.6f}, std={np.std(neg_d2p):.6f}")
    print(f"    Mann-Whitney U={u_stat:.1f}, p={p_val:.4e}")
    print()

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Histogram
    axes[0].hist(pos_d2p, bins=20, alpha=0.6, label='Positive', color='green', density=True)
    axes[0].hist(neg_d2p, bins=20, alpha=0.6, label='Negative', color='red', density=True)
    axes[0].set_xlabel('Self-Attention Score (Drug->Protein)')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Self-Attention Distribution\nMann-Whitney p={p_val:.4e}')
    axes[0].legend()

    # ROC Curve
    fpr, tpr, _ = roc_curve(ground_truths, d2p_scores)
    axes[1].plot(fpr, tpr, color='steelblue', linewidth=2, label=f'Drug->Protein (AUC={auc_d2p:.4f})')
    fpr2, tpr2, _ = roc_curve(ground_truths, p2d_scores)
    axes[1].plot(fpr2, tpr2, color='coral', linewidth=2, label=f'Protein->Drug (AUC={auc_p2d:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC: Self-Attention as Predictor')
    axes[1].legend()

    # Prediction vs Self-Attention scatter
    predictions = [r['prediction'] for r in hl_results]
    colors = ['green' if gt == 1 else 'red' for gt in ground_truths]
    axes[2].scatter(d2p_scores, predictions, c=colors, alpha=0.6, s=40)
    rho, p_rho = spearmanr(d2p_scores, predictions)
    axes[2].set_xlabel('Self-Attention Drug->Protein')
    axes[2].set_ylabel('Model Prediction Score')
    axes[2].set_title(f'Self-Attention vs Prediction\nSpearman rho={rho:.4f}, p={p_rho:.4e}')
    axes[2].legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Positive'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Negative')
    ])

    plt.suptitle('HL-2: Self-Attention Discriminative Power Validation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'self_attention_validation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    return {
        'auc_d2p': auc_d2p,
        'auc_p2d': auc_p2d,
        'mann_whitney_p': p_val,
        'attn_pred_rho': rho,
        'attn_pred_p': p_rho,
    }


# =====================================================================
# HL-7: PREDICTION CALIBRATION
# =====================================================================

def validate_prediction_calibration(hl_results, save_dir):
    """Check prediction score calibration and quality."""
    print("=" * 70)
    print("HL-7: PREDICTION CALIBRATION")
    print("=" * 70)
    print()

    predictions = np.array([r['prediction'] for r in hl_results])
    ground_truths = np.array([r['ground_truth'] for r in hl_results])

    # AUC on this subset
    auc = roc_auc_score(ground_truths, predictions)
    print(f"  AUC on 100-sample subset: {auc:.4f}")

    # Accuracy at threshold=0.5
    pred_labels = (predictions >= 0.5).astype(int)
    accuracy = np.mean(pred_labels == ground_truths)
    print(f"  Accuracy (threshold=0.5): {accuracy:.4f}")

    # Calibration curve (reliability diagram)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_freqs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_freqs.append(np.mean(ground_truths[mask]))
            bin_counts.append(np.sum(mask))

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Calibration plot
    axes[0].plot(bin_centers, bin_freqs, 'o-', color='steelblue', linewidth=2, markersize=8)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Observed Frequency')
    axes[0].set_title(f'Calibration Curve (Reliability Diagram)\nAUC={auc:.4f}')
    axes[0].legend()
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)

    # Prediction distribution
    pos_preds = predictions[ground_truths == 1]
    neg_preds = predictions[ground_truths == 0]
    axes[1].hist(pos_preds, bins=20, alpha=0.6, label=f'Positive (n={len(pos_preds)})', color='green', density=True)
    axes[1].hist(neg_preds, bins=20, alpha=0.6, label=f'Negative (n={len(neg_preds)})', color='red', density=True)
    axes[1].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Prediction Score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Prediction Distribution by Label')
    axes[1].legend()

    # ROC curve
    fpr, tpr, _ = roc_curve(ground_truths, predictions)
    axes[2].plot(fpr, tpr, color='steelblue', linewidth=2, label=f'AUC={auc:.4f}')
    axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title(f'ROC Curve (100-sample subset)')
    axes[2].legend()

    plt.suptitle('HL-7: Prediction Calibration & Quality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'prediction_calibration.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    return {'auc': auc, 'accuracy': accuracy}


# =====================================================================
# MAIN
# =====================================================================

def main():
    print()
    print("=" * 70)
    print("VALIDATION: ATTENTION ANALYSIS (HL-1, HL-2, HL-7)")
    print("=" * 70)
    print()

    SAVE_DIR = os.path.join('results', 'validation', 'attention')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load saved high-level results
    print("Loading saved prediction-level results...")
    hl_results = torch.load(
        os.path.join('results', 'explainability_highlevel', 'all_results.pt'),
        map_location='cpu'
    )
    print(f"  Loaded {len(hl_results)} high-level results")
    print()

    # --- HL-1 ---
    alpha_results = validate_alpha_distribution(hl_results, SAVE_DIR)
    print()

    # --- HL-2 ---
    attn_results = validate_self_attention(hl_results, SAVE_DIR)
    print()

    # --- HL-7 ---
    calibration_results = validate_prediction_calibration(hl_results, SAVE_DIR)
    print()

    # --- Save all results ---
    torch.save({
        'alpha': alpha_results,
        'attention': attn_results,
        'calibration': calibration_results,
    }, os.path.join(SAVE_DIR, 'attention_validation_results.pt'))

    # --- Write summary report ---
    lines = []
    lines.append("=" * 70)
    lines.append("ATTENTION & CALIBRATION VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("HL-1: ALPHA DISTRIBUTION")
    lines.append(f"  Drug vs Protein alpha:         p={alpha_results['drug_vs_protein_p']:.4e}")
    lines.append(f"  Pos vs Neg Drug alpha:         p={alpha_results['pos_vs_neg_drug_p']:.4e}")
    lines.append(f"  Drug alpha <-> Prediction:     rho={alpha_results['drug_alpha_pred_rho']:.4f}")
    lines.append("")
    lines.append("HL-2: SELF-ATTENTION DISCRIMINATIVE POWER")
    lines.append(f"  Drug->Protein Attention AUC:   {attn_results['auc_d2p']:.4f}")
    lines.append(f"  Protein->Drug Attention AUC:   {attn_results['auc_p2d']:.4f}")
    lines.append(f"  Mann-Whitney (pos vs neg):     p={attn_results['mann_whitney_p']:.4e}")
    lines.append("")
    lines.append("HL-7: PREDICTION CALIBRATION")
    lines.append(f"  100-sample AUC:                {calibration_results['auc']:.4f}")
    lines.append(f"  Accuracy (threshold=0.5):      {calibration_results['accuracy']:.4f}")
    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    report_path = os.path.join(SAVE_DIR, 'attention_validation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    print()
    print("=" * 70)
    print("ATTENTION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
