"""
=================================================================
VALIDATION: DECODER MASKING & IG CONVERGENCE (HL-3, HL-4)
=================================================================

HL-3: Decoder Dimension Masking
  - Zero out the top-10 most important z_hat dimensions
  - Compare prediction drop vs zeroing random-10 dimensions
  - If the top dims are truly important, masking them should
    cause a larger prediction drop than masking random dims.

HL-4: IG Convergence (Completeness Axiom)
  - For each sample, re-run IG with return_convergence_delta=True
  - The convergence delta should be close to 0 (completeness axiom:
    sum of attributions = F(input) - F(baseline))
  - Small delta = IG is reliable; large delta = numerical issues

Uses saved results + re-runs decoder for masking test.
=================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wilcoxon

# Local imports
from model import H2GNN
import opt


# =====================================================================
# HL-3: DECODER DIMENSION MASKING
# =====================================================================

def hl3_decoder_masking(hl_results, h2gnn, features, adj, nb_drugs, save_dir):
    """
    For each sample, zero out the top-10 z_hat dims (identified by Level 2)
    and compare prediction drop vs zeroing random-10 dims.
    """
    print("=" * 70)
    print("HL-3: DECODER DIMENSION MASKING")
    print("=" * 70)
    print()

    K = 10
    N_RANDOM = 20
    np.random.seed(42)

    h2gnn.eval()
    device = features.device

    top_drops = []
    rand_drops = []
    original_preds = []

    for s_idx, result in enumerate(hl_results):
        drug_idx = result.get('drug_idx')
        protein_idx = result.get('protein_idx')

        # If indices not directly stored, we need to find them
        if drug_idx is None:
            drug_id = result['drug_id']
            protein_id = result['protein_id']
            # We'll skip samples without indices
            continue

        pro_feat_idx = nb_drugs + protein_idx

        # Get z_hat from original forward pass
        with torch.no_grad():
            x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde, alpha = h2gnn(features, adj)

            # Original prediction
            orig_pred = torch.sigmoid(
                torch.dot(z_hat[drug_idx], z_hat[pro_feat_idx])
            ).item()

            # Get top-K dims from Level 2 attribution
            l2 = result['level2']
            attr_drug = l2['attr_drug_dims']
            attr_protein = l2['attr_protein_dims']
            if isinstance(attr_drug, torch.Tensor):
                attr_drug = attr_drug.numpy()
            if isinstance(attr_protein, torch.Tensor):
                attr_protein = attr_protein.numpy()

            # Combined attribution across drug+protein
            combined_attr = np.abs(attr_drug) + np.abs(attr_protein)
            top_dims = np.argsort(combined_attr)[::-1][:K].copy()

            # Mask top-K dims
            z_hat_masked = z_hat.clone()
            z_hat_masked[drug_idx, top_dims] = 0
            z_hat_masked[pro_feat_idx, top_dims] = 0
            top_pred = torch.sigmoid(
                torch.dot(z_hat_masked[drug_idx], z_hat_masked[pro_feat_idx])
            ).item()

            # Mask random-K dims (average over N_RANDOM trials)
            rand_preds = []
            for _ in range(N_RANDOM):
                rand_dims = np.random.choice(z_hat.shape[1], K, replace=False)
                z_hat_rand = z_hat.clone()
                z_hat_rand[drug_idx, rand_dims] = 0
                z_hat_rand[pro_feat_idx, rand_dims] = 0
                r_pred = torch.sigmoid(
                    torch.dot(z_hat_rand[drug_idx], z_hat_rand[pro_feat_idx])
                ).item()
                rand_preds.append(r_pred)

            rand_pred_mean = np.mean(rand_preds)

            top_drop = abs(orig_pred - top_pred)
            rand_drop = abs(orig_pred - rand_pred_mean)

            top_drops.append(top_drop)
            rand_drops.append(rand_drop)
            original_preds.append(orig_pred)

        if s_idx < 5 or s_idx % 20 == 0:
            print(f"  Sample {s_idx:3d}: orig={orig_pred:.4f}, "
                  f"top-{K} masked={top_pred:.4f} (drop={top_drop:.4f}), "
                  f"rand-{K} masked={rand_pred_mean:.4f} (drop={rand_drop:.4f})")

    print()

    if not top_drops:
        print("  ERROR: No samples processed. Check if drug_idx/protein_idx are stored.")
        return {}

    # Statistical test
    try:
        w_stat, w_p = wilcoxon(top_drops, rand_drops, alternative='greater')
    except Exception:
        w_stat, w_p = 0, 1.0

    ratio = np.mean(top_drops) / np.mean(rand_drops) if np.mean(rand_drops) > 0 else float('inf')
    sig = '***' if w_p < 0.001 else '**' if w_p < 0.01 else '*' if w_p < 0.05 else 'n.s.'

    print(f"  Mean top-{K} prediction drop:   {np.mean(top_drops):.6f}")
    print(f"  Mean rand-{K} prediction drop:  {np.mean(rand_drops):.6f}")
    print(f"  Ratio (top/random):             {ratio:.2f}x")
    print(f"  Wilcoxon signed-rank p:         {w_p:.4e} [{sig}]")
    print()

    return {
        'top_drops': top_drops,
        'rand_drops': rand_drops,
        'original_preds': original_preds,
        'ratio': ratio,
        'wilcoxon_p': w_p,
        'K': K,
    }


# =====================================================================
# HL-4: IG CONVERGENCE DELTA
# =====================================================================

def hl4_ig_convergence(hl_results, h2gnn, features, adj, nb_drugs, save_dir):
    """
    Re-run IG on a subset of samples with return_convergence_delta=True.
    Check if the completeness axiom holds (delta close to 0).
    """
    from captum.attr import IntegratedGradients

    print("=" * 70)
    print("HL-4: IG CONVERGENCE (COMPLETENESS AXIOM)")
    print("=" * 70)
    print()

    h2gnn.eval()
    device = features.device

    # We'll test convergence on Level 3 (z_tilde -> decoder)
    # Use a subset for speed
    N_TEST = min(20, len(hl_results))

    deltas_l3 = []
    deltas_l4 = []
    deltas_l5 = []

    # --- Level 3: z_tilde -> decoder ---
    print("  Testing Level 3 (z_tilde -> decoder) convergence...")
    from explainability_highlevel import H2GNNDecoderWrapper

    for s_idx in range(N_TEST):
        result = hl_results[s_idx]
        drug_idx = result.get('drug_idx')
        protein_idx = result.get('protein_idx')
        if drug_idx is None:
            continue

        with torch.no_grad():
            _, _, _, _, _, z_tilde_orig, _ = h2gnn(features, adj)

        wrapper = H2GNNDecoderWrapper(
            h2gnn.gae.decoder, adj, drug_idx, protein_idx, nb_drugs
        )

        z_tilde_input = z_tilde_orig.unsqueeze(0).clone().detach().requires_grad_(True)
        baseline = torch.zeros_like(z_tilde_input)

        ig = IntegratedGradients(wrapper)
        attributions, delta = ig.attribute(
            z_tilde_input,
            baselines=baseline,
            n_steps=200,
            method='gausslegendre',
            return_convergence_delta=True
        )

        delta_val = abs(delta.item())
        deltas_l3.append(delta_val)

        if s_idx < 5:
            print(f"    Sample {s_idx}: convergence delta = {delta_val:.6e}")

    print(f"  Level 3 deltas: mean={np.mean(deltas_l3):.6e}, "
          f"max={np.max(deltas_l3):.6e}, median={np.median(deltas_l3):.6e}")
    print()

    # --- Level 4: AE convergence ---
    print("  Testing Level 4 (AE features -> encoder) convergence...")
    from explainability_highlevel import AEPairWrapper

    for s_idx in range(N_TEST):
        result = hl_results[s_idx]
        drug_idx = result.get('drug_idx')
        protein_idx = result.get('protein_idx')
        if drug_idx is None:
            continue

        wrapper_ae = AEPairWrapper(
            h2gnn.ae.encoder, drug_idx, protein_idx, nb_drugs
        )

        features_input = features.unsqueeze(0).clone().detach().requires_grad_(True)
        baseline_ae = torch.zeros_like(features_input)

        ig_ae = IntegratedGradients(wrapper_ae)
        attrs_ae, delta_ae = ig_ae.attribute(
            features_input,
            baselines=baseline_ae,
            n_steps=200,
            method='gausslegendre',
            return_convergence_delta=True
        )

        delta_val_ae = abs(delta_ae.item())
        deltas_l4.append(delta_val_ae)

        if s_idx < 5:
            print(f"    Sample {s_idx}: convergence delta = {delta_val_ae:.6e}")

    print(f"  Level 4 deltas: mean={np.mean(deltas_l4):.6e}, "
          f"max={np.max(deltas_l4):.6e}, median={np.median(deltas_l4):.6e}")
    print()

    # --- Level 5: IGAE convergence ---
    print("  Testing Level 5 (IGAE features -> encoder) convergence...")
    from explainability_highlevel import IGAEPairWrapper

    for s_idx in range(N_TEST):
        result = hl_results[s_idx]
        drug_idx = result.get('drug_idx')
        protein_idx = result.get('protein_idx')
        if drug_idx is None:
            continue

        wrapper_igae = IGAEPairWrapper(
            h2gnn.gae.encoder, adj, drug_idx, protein_idx, nb_drugs
        )

        features_input = features.unsqueeze(0).clone().detach().requires_grad_(True)
        baseline_igae = torch.zeros_like(features_input)

        ig_igae = IntegratedGradients(wrapper_igae)
        attrs_igae, delta_igae = ig_igae.attribute(
            features_input,
            baselines=baseline_igae,
            n_steps=200,
            method='gausslegendre',
            return_convergence_delta=True
        )

        delta_val_igae = abs(delta_igae.item())
        deltas_l5.append(delta_val_igae)

        if s_idx < 5:
            print(f"    Sample {s_idx}: convergence delta = {delta_val_igae:.6e}")

    print(f"  Level 5 deltas: mean={np.mean(deltas_l5):.6e}, "
          f"max={np.max(deltas_l5):.6e}, median={np.median(deltas_l5):.6e}")
    print()

    # Interpretation
    all_deltas = deltas_l3 + deltas_l4 + deltas_l5
    overall_mean = np.mean(all_deltas) if all_deltas else 0
    print(f"  Overall mean convergence delta: {overall_mean:.6e}")
    if overall_mean < 0.01:
        print("  -> EXCELLENT: IG completeness axiom holds tightly")
    elif overall_mean < 0.1:
        print("  -> GOOD: IG convergence is acceptable")
    elif overall_mean < 1.0:
        print("  -> MODERATE: Some numerical drift, consider more IG steps")
    else:
        print("  -> POOR: IG did not converge well, results may be unreliable")
    print()

    return {
        'deltas_l3': deltas_l3,
        'deltas_l4': deltas_l4,
        'deltas_l5': deltas_l5,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    print()
    print("=" * 70)
    print("VALIDATION: DECODER MASKING (HL-3) & IG CONVERGENCE (HL-4)")
    print("=" * 70)
    print()

    SAVE_DIR = os.path.join('results', 'validation', 'decoder_convergence')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Load saved results ---
    print("Loading saved results...")
    hl_results = torch.load(
        os.path.join('results', 'explainability_highlevel', 'all_results.pt'),
        map_location='cpu'
    )
    print(f"  Loaded {len(hl_results)} high-level results")

    # Check if drug_idx/protein_idx are stored
    sample0 = hl_results[0]
    has_indices = 'drug_idx' in sample0
    if not has_indices:
        print("  drug_idx not found in results. Reconstructing from data...")
        data_ckpt = torch.load('checkpoints/data_kiba.pt', map_location='cpu')
        data_new = data_ckpt['data_new']

        # Build maps
        drugid = sorted(set([item[0] for item in data_new]))
        proteinid = sorted(set([item[1] for item in data_new]))
        drugmap = {did: idx for idx, did in enumerate(drugid)}
        proteinmap = {pid: idx for idx, pid in enumerate(proteinid)}

        for r in hl_results:
            r['drug_idx'] = drugmap.get(r['drug_id'])
            r['protein_idx'] = proteinmap.get(r['protein_id'])
        print("  Reconstructed drug_idx and protein_idx for all samples")

    # --- Load model ---
    print("Loading model...")
    data_ckpt = torch.load('checkpoints/data_kiba.pt', map_location='cpu')
    nb_drugs = data_ckpt['nb_drugs']
    nb_proteins = data_ckpt['nb_proteins']
    nb_all = nb_drugs + nb_proteins
    full_features = data_ckpt['features']
    full_adj = data_ckpt['adj']

    device = torch.device('cpu')
    h2gnn = H2GNN(n_node=nb_all).to(device)

    h2gnn_state = torch.load('checkpoints/h2gnn_kiba.pt', map_location='cpu')
    h2gnn.load_state_dict(h2gnn_state)
    print(f"  Model loaded. nb_drugs={nb_drugs}, nb_all={nb_all}")

    features = full_features.to(device)
    adj = full_adj.to(device)
    print()

    # --- Run HL-3 ---
    hl3_results = hl3_decoder_masking(hl_results, h2gnn, features, adj, nb_drugs, SAVE_DIR)

    # --- Run HL-4 ---
    hl4_results = hl4_ig_convergence(hl_results, h2gnn, features, adj, nb_drugs, SAVE_DIR)

    # ===================================================================
    # VISUALIZATIONS
    # ===================================================================
    print("Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Plot 1: HL-3 Masking comparison ---
    if hl3_results and hl3_results.get('top_drops'):
        K = hl3_results['K']
        bp = axes[0, 0].boxplot(
            [hl3_results['top_drops'], hl3_results['rand_drops']],
            labels=[f'Top-{K} Masked', f'Random-{K} Masked'],
            patch_artist=True
        )
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('gray')
        bp['boxes'][1].set_alpha(0.6)
        ratio = hl3_results['ratio']
        p_val = hl3_results['wilcoxon_p']
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        axes[0, 0].set_ylabel('Prediction Drop (|orig - masked|)')
        axes[0, 0].set_title(f'HL-3: Decoder Dimension Masking\n'
                              f'Ratio={ratio:.2f}x, p={p_val:.2e} [{sig}]')

    # --- Plot 2: HL-3 Scatter (top drop vs random drop) ---
    if hl3_results and hl3_results.get('top_drops'):
        axes[0, 1].scatter(hl3_results['rand_drops'], hl3_results['top_drops'],
                           alpha=0.6, color='steelblue', s=40)
        max_val = max(max(hl3_results['rand_drops']), max(hl3_results['top_drops'])) * 1.1
        axes[0, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x (equal drop)')
        axes[0, 1].set_xlabel('Random-K Prediction Drop')
        axes[0, 1].set_ylabel('Top-K Prediction Drop')
        axes[0, 1].set_title('HL-3: Top-K vs Random-K Prediction Drop')
        axes[0, 1].legend()

    # --- Plot 3: HL-4 Convergence deltas ---
    if hl4_results and hl4_results.get('deltas_l3'):
        all_labels = []
        all_deltas = []
        for label, key in [('Level 3', 'deltas_l3'), ('Level 4', 'deltas_l4'), ('Level 5', 'deltas_l5')]:
            vals = hl4_results.get(key, [])
            if vals:
                all_labels.append(label)
                all_deltas.append(vals)

        if all_deltas:
            bp2 = axes[1, 0].boxplot(all_deltas, labels=all_labels, patch_artist=True)
            colors_bp = ['steelblue', 'coral', 'seagreen']
            for patch, c in zip(bp2['boxes'], colors_bp[:len(bp2['boxes'])]):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            axes[1, 0].set_ylabel('|Convergence Delta|')
            axes[1, 0].set_title('HL-4: IG Convergence Delta by Level')
            axes[1, 0].set_yscale('log')

    # --- Plot 4: Summary table ---
    axes[1, 1].axis('off')
    table_data = [['Metric', 'Value', 'Verdict']]

    if hl3_results and hl3_results.get('top_drops'):
        sig = '***' if hl3_results['wilcoxon_p'] < 0.001 else '**' if hl3_results['wilcoxon_p'] < 0.01 else '*' if hl3_results['wilcoxon_p'] < 0.05 else 'n.s.'
        table_data.append(['HL-3 Ratio', f"{hl3_results['ratio']:.2f}x", sig])
        table_data.append(['HL-3 p-value', f"{hl3_results['wilcoxon_p']:.2e}", ''])

    if hl4_results:
        for label, key in [('L3 Delta', 'deltas_l3'), ('L4 Delta', 'deltas_l4'), ('L5 Delta', 'deltas_l5')]:
            vals = hl4_results.get(key, [])
            if vals:
                mean_d = np.mean(vals)
                verdict = 'Excellent' if mean_d < 0.01 else 'Good' if mean_d < 0.1 else 'Moderate'
                table_data.append([label, f"{mean_d:.2e}", verdict])

    table = axes[1, 1].table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)
    axes[1, 1].set_title('Results Summary', fontsize=14, fontweight='bold')

    plt.suptitle('HL-3: Decoder Masking & HL-4: IG Convergence', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, 'decoder_convergence.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # ===================================================================
    # SAVE & REPORT
    # ===================================================================
    torch.save({
        'hl3': hl3_results,
        'hl4': hl4_results,
    }, os.path.join(SAVE_DIR, 'decoder_convergence_results.pt'))

    lines = []
    lines.append("=" * 70)
    lines.append("HL-3 & HL-4: DECODER MASKING & IG CONVERGENCE REPORT")
    lines.append("=" * 70)
    lines.append("")

    if hl3_results and hl3_results.get('top_drops'):
        sig = '***' if hl3_results['wilcoxon_p'] < 0.001 else '**' if hl3_results['wilcoxon_p'] < 0.01 else '*' if hl3_results['wilcoxon_p'] < 0.05 else 'n.s.'
        lines.append("HL-3: DECODER DIMENSION MASKING")
        lines.append(f"  Top-{hl3_results['K']} mean drop:    {np.mean(hl3_results['top_drops']):.6f}")
        lines.append(f"  Random-{hl3_results['K']} mean drop: {np.mean(hl3_results['rand_drops']):.6f}")
        lines.append(f"  Ratio:                {hl3_results['ratio']:.2f}x")
        lines.append(f"  Wilcoxon p:           {hl3_results['wilcoxon_p']:.4e} [{sig}]")
        lines.append("")

    if hl4_results:
        lines.append("HL-4: IG CONVERGENCE")
        for label, key in [('Level 3', 'deltas_l3'), ('Level 4', 'deltas_l4'), ('Level 5', 'deltas_l5')]:
            vals = hl4_results.get(key, [])
            if vals:
                lines.append(f"  {label}: mean={np.mean(vals):.6e}, "
                             f"max={np.max(vals):.6e}, median={np.median(vals):.6e}")
        lines.append("")

    lines.append("=" * 70)

    report = "\n".join(lines)
    report_path = os.path.join(SAVE_DIR, 'decoder_convergence_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    print()
    print("=" * 70)
    print("HL-3 & HL-4 VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
