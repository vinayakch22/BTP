"""
=================================================================
VALIDATION: PERTURBATION FAITHFULNESS TEST (LL-1 & LL-2)
=================================================================

Validates that Integrated Gradients attributions from the low-level
GNNNet explainability pipeline are *faithful* — i.e., the features
identified as important actually matter for the model's computation.

Method:
  1. For each sample, identify the top-K atoms/residues by IG importance.
  2. Zero out their features and recompute the GNNNet branch embedding.
  3. Measure the L2-norm change from the original embedding.
  4. Repeat with RANDOM K atoms/residues (averaged over 10 trials).
  5. If attributions are faithful, top-K masking should cause a LARGER
     embedding change than random-K masking.

Statistical Test: Paired t-test (top-K drop vs random-K drop per sample).
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

from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool as gep

# Local imports
from NodeRepresentation import GNNNet
from model import H2GNN
from data_load import dataload
import opt

# Re-use helpers from the explainability pipeline
from explainability_lowlevel import (
    build_drug_protein_maps,
    get_interaction_pairs,
    get_drug_graph_data,
    get_protein_graph_data,
    DrugGATWrapper,
    ProteinGATWrapper,
)


# =====================================================================
# CORE PERTURBATION FUNCTIONS
# =====================================================================

def compute_drug_embedding_norm(gnnnet, drug_data, device):
    """Compute the L2 norm of the drug branch embedding (no dropout)."""
    gnnnet.eval()
    with torch.no_grad():
        x = drug_data.x.to(device)
        edge_index = drug_data.edge_index.to(device)
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)

        x_out = F.relu(gnnnet.mol_conv1(x, edge_index))
        x_out = F.relu(gnnnet.mol_conv2(x_out, edge_index))
        x_out = F.relu(gnnnet.mol_conv3(x_out, edge_index))
        x_out = gep(x_out, batch)
        x_out = F.relu(gnnnet.mol_fc_g1(x_out))
        x_out = gnnnet.mol_fc_g2(x_out)

    return torch.norm(x_out).item(), x_out.cpu().numpy()


def compute_protein_embedding_norm(gnnnet, pro_data, device):
    """Compute the L2 norm of the protein branch embedding (no dropout)."""
    gnnnet.eval()
    with torch.no_grad():
        xt = pro_data.x.to(device)
        edge_index = pro_data.edge_index.to(device)
        batch = torch.zeros(xt.shape[0], dtype=torch.long, device=device)

        xt_out = F.relu(gnnnet.pro_conv1(xt, edge_index))
        xt_out = F.relu(gnnnet.pro_conv2(xt_out, edge_index))
        xt_out = F.relu(gnnnet.pro_conv3(xt_out, edge_index))
        xt_out = gep(xt_out, batch)
        xt_out = F.relu(gnnnet.pro_fc_g1(xt_out))
        xt_out = gnnnet.pro_fc_g2(xt_out)

    return torch.norm(xt_out).item(), xt_out.cpu().numpy()


def perturb_and_measure_drug(gnnnet, drug_data, mask_indices, device):
    """
    Zero out features at mask_indices in the drug graph, recompute
    the embedding, and return its L2 norm.
    """
    perturbed_x = drug_data.x.clone()
    for idx in mask_indices:
        if idx < perturbed_x.shape[0]:
            perturbed_x[idx] = 0.0

    perturbed_data = Data(
        x=perturbed_x,
        edge_index=drug_data.edge_index,
        batch=drug_data.batch if hasattr(drug_data, 'batch') and drug_data.batch is not None
              else torch.zeros(perturbed_x.shape[0], dtype=torch.long)
    )
    norm_val, emb = compute_drug_embedding_norm(gnnnet, perturbed_data, device)
    return norm_val, emb


def perturb_and_measure_protein(gnnnet, pro_data, mask_indices, device):
    """
    Zero out features at mask_indices in the protein graph, recompute
    the embedding, and return its L2 norm.
    """
    perturbed_x = pro_data.x.clone()
    for idx in mask_indices:
        if idx < perturbed_x.shape[0]:
            perturbed_x[idx] = 0.0

    perturbed_data = Data(
        x=perturbed_x,
        edge_index=pro_data.edge_index,
        batch=pro_data.batch if hasattr(pro_data, 'batch') and pro_data.batch is not None
              else torch.zeros(perturbed_x.shape[0], dtype=torch.long)
    )
    norm_val, emb = compute_protein_embedding_norm(gnnnet, perturbed_data, device)
    return norm_val, emb


def run_perturbation_test_drug(gnnnet, drug_data, node_importance, device,
                                K_values=[3, 5, 10], n_random_trials=10):
    """
    Run perturbation test for a single drug.
    Returns dict of {K: (top_k_drop, random_k_drop_mean, random_k_drop_std)}
    """
    num_atoms = drug_data.x.shape[0]
    original_norm, original_emb = compute_drug_embedding_norm(gnnnet, drug_data, device)

    results = {}
    for K in K_values:
        if K >= num_atoms:
            K_actual = max(1, num_atoms - 1)
        else:
            K_actual = K

        # Top-K by importance
        sorted_idx = np.argsort(node_importance)[::-1][:K_actual]
        top_k_norm, top_k_emb = perturb_and_measure_drug(
            gnnnet, drug_data, sorted_idx.tolist(), device
        )
        top_k_drop = np.linalg.norm(original_emb - top_k_emb)

        # Random-K (average over n_random_trials)
        random_drops = []
        for _ in range(n_random_trials):
            random_idx = np.random.choice(num_atoms, size=K_actual, replace=False)
            rand_norm, rand_emb = perturb_and_measure_drug(
                gnnnet, drug_data, random_idx.tolist(), device
            )
            random_drops.append(np.linalg.norm(original_emb - rand_emb))

        results[K] = {
            'top_k_drop': top_k_drop,
            'random_k_mean': np.mean(random_drops),
            'random_k_std': np.std(random_drops),
            'original_norm': original_norm,
        }

    return results


def run_perturbation_test_protein(gnnnet, pro_data, node_importance, device,
                                   K_values=[5, 10, 20], n_random_trials=10):
    """
    Run perturbation test for a single protein.
    """
    num_residues = pro_data.x.shape[0]
    original_norm, original_emb = compute_protein_embedding_norm(gnnnet, pro_data, device)

    results = {}
    for K in K_values:
        if K >= num_residues:
            K_actual = max(1, num_residues - 1)
        else:
            K_actual = K

        sorted_idx = np.argsort(node_importance)[::-1][:K_actual]
        top_k_norm, top_k_emb = perturb_and_measure_protein(
            gnnnet, pro_data, sorted_idx.tolist(), device
        )
        top_k_drop = np.linalg.norm(original_emb - top_k_emb)

        random_drops = []
        for _ in range(n_random_trials):
            random_idx = np.random.choice(num_residues, size=K_actual, replace=False)
            rand_norm, rand_emb = perturb_and_measure_protein(
                gnnnet, pro_data, random_idx.tolist(), device
            )
            random_drops.append(np.linalg.norm(original_emb - rand_emb))

        results[K] = {
            'top_k_drop': top_k_drop,
            'random_k_mean': np.mean(random_drops),
            'random_k_std': np.std(random_drops),
            'original_norm': original_norm,
        }

    return results


# =====================================================================
# VISUALIZATION
# =====================================================================

def plot_perturbation_results(all_drug_results, all_protein_results,
                               drug_K_values, protein_K_values, save_dir):
    """Generate publication-quality bar charts and statistical summaries."""

    # --- Drug Perturbation Results ---
    fig, axes = plt.subplots(1, len(drug_K_values), figsize=(5 * len(drug_K_values), 6))
    if len(drug_K_values) == 1:
        axes = [axes]

    for ax_idx, K in enumerate(drug_K_values):
        top_drops = [r[K]['top_k_drop'] for r in all_drug_results if K in r]
        rand_drops = [r[K]['random_k_mean'] for r in all_drug_results if K in r]

        if len(top_drops) < 2:
            continue

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(top_drops, rand_drops)

        ax = axes[ax_idx]
        x_pos = [0, 1]
        means = [np.mean(top_drops), np.mean(rand_drops)]
        stds = [np.std(top_drops), np.std(rand_drops)]
        bars = ax.bar(x_pos, means, yerr=stds, capsize=8,
                      color=['#d62728', '#2ca02c'], alpha=0.8, width=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Top-{K}\n(Important)', f'Random-{K}'])
        ax.set_ylabel('Embedding L2 Distance Change')
        ax.set_title(f'Drug Atoms (K={K})\np={p_value:.4e}, t={t_stat:.2f}')

        # Significance markers
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = 'n.s.'

        max_y = max(means[0] + stds[0], means[1] + stds[1])
        ax.annotate(sig_text, xy=(0.5, max_y * 1.05),
                    fontsize=16, ha='center', fontweight='bold')

    plt.suptitle('LL-1: Drug Atom Perturbation Faithfulness Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'drug_perturbation_faithfulness.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # --- Protein Perturbation Results ---
    fig, axes = plt.subplots(1, len(protein_K_values), figsize=(5 * len(protein_K_values), 6))
    if len(protein_K_values) == 1:
        axes = [axes]

    for ax_idx, K in enumerate(protein_K_values):
        top_drops = [r[K]['top_k_drop'] for r in all_protein_results if K in r]
        rand_drops = [r[K]['random_k_mean'] for r in all_protein_results if K in r]

        if len(top_drops) < 2:
            continue

        t_stat, p_value = stats.ttest_rel(top_drops, rand_drops)

        ax = axes[ax_idx]
        x_pos = [0, 1]
        means = [np.mean(top_drops), np.mean(rand_drops)]
        stds = [np.std(top_drops), np.std(rand_drops)]
        bars = ax.bar(x_pos, means, yerr=stds, capsize=8,
                      color=['#d62728', '#2ca02c'], alpha=0.8, width=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Top-{K}\n(Important)', f'Random-{K}'])
        ax.set_ylabel('Embedding L2 Distance Change')
        ax.set_title(f'Protein Residues (K={K})\np={p_value:.4e}, t={t_stat:.2f}')

        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = 'n.s.'

        max_y = max(means[0] + stds[0], means[1] + stds[1])
        ax.annotate(sig_text, xy=(0.5, max_y * 1.05),
                    fontsize=16, ha='center', fontweight='bold')

    plt.suptitle('LL-2: Protein Residue Perturbation Faithfulness Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'protein_perturbation_faithfulness.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # --- Combined per-sample scatter plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Drug scatter (use middle K value)
    K_drug = drug_K_values[len(drug_K_values) // 2]
    top_d = [r[K_drug]['top_k_drop'] for r in all_drug_results if K_drug in r]
    rand_d = [r[K_drug]['random_k_mean'] for r in all_drug_results if K_drug in r]
    axes[0].scatter(rand_d, top_d, alpha=0.6, c='steelblue', s=40)
    max_val = max(max(top_d), max(rand_d)) * 1.1
    axes[0].plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal line')
    axes[0].set_xlabel(f'Random-{K_drug} Embedding Change')
    axes[0].set_ylabel(f'Top-{K_drug} Embedding Change')
    axes[0].set_title(f'Drug: Top-{K_drug} vs Random-{K_drug} (per sample)')
    axes[0].legend()

    # Protein scatter
    K_pro = protein_K_values[len(protein_K_values) // 2]
    top_p = [r[K_pro]['top_k_drop'] for r in all_protein_results if K_pro in r]
    rand_p = [r[K_pro]['random_k_mean'] for r in all_protein_results if K_pro in r]
    axes[1].scatter(rand_p, top_p, alpha=0.6, c='coral', s=40)
    max_val = max(max(top_p), max(rand_p)) * 1.1
    axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal line')
    axes[1].set_xlabel(f'Random-{K_pro} Embedding Change')
    axes[1].set_ylabel(f'Top-{K_pro} Embedding Change')
    axes[1].set_title(f'Protein: Top-{K_pro} vs Random-{K_pro} (per sample)')
    axes[1].legend()

    plt.suptitle('Per-Sample Perturbation: Top-K vs Random-K', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'perturbation_scatter.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")


def print_statistical_summary(all_drug_results, all_protein_results,
                               drug_K_values, protein_K_values, save_dir):
    """Print and save a comprehensive statistical summary table."""

    lines = []
    lines.append("=" * 70)
    lines.append("PERTURBATION FAITHFULNESS TEST — STATISTICAL SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # Drug results
    lines.append("DRUG ATOM PERTURBATION (LL-1)")
    lines.append("-" * 50)
    lines.append(f"{'K':>4s}  {'Top-K Mean':>12s}  {'Rand-K Mean':>12s}  {'t-stat':>8s}  {'p-value':>12s}  {'Sig':>5s}  {'Ratio':>6s}")
    lines.append("-" * 50)
    for K in drug_K_values:
        top_drops = [r[K]['top_k_drop'] for r in all_drug_results if K in r]
        rand_drops = [r[K]['random_k_mean'] for r in all_drug_results if K in r]
        if len(top_drops) < 2:
            continue
        t_stat, p_value = stats.ttest_rel(top_drops, rand_drops)
        ratio = np.mean(top_drops) / (np.mean(rand_drops) + 1e-10)
        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'
        lines.append(f"{K:4d}  {np.mean(top_drops):12.6f}  {np.mean(rand_drops):12.6f}  "
                      f"{t_stat:8.2f}  {p_value:12.4e}  {sig:>5s}  {ratio:6.2f}x")

    lines.append("")

    # Protein results
    lines.append("PROTEIN RESIDUE PERTURBATION (LL-2)")
    lines.append("-" * 50)
    lines.append(f"{'K':>4s}  {'Top-K Mean':>12s}  {'Rand-K Mean':>12s}  {'t-stat':>8s}  {'p-value':>12s}  {'Sig':>5s}  {'Ratio':>6s}")
    lines.append("-" * 50)
    for K in protein_K_values:
        top_drops = [r[K]['top_k_drop'] for r in all_protein_results if K in r]
        rand_drops = [r[K]['random_k_mean'] for r in all_protein_results if K in r]
        if len(top_drops) < 2:
            continue
        t_stat, p_value = stats.ttest_rel(top_drops, rand_drops)
        ratio = np.mean(top_drops) / (np.mean(rand_drops) + 1e-10)
        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'
        lines.append(f"{K:4d}  {np.mean(top_drops):12.6f}  {np.mean(rand_drops):12.6f}  "
                      f"{t_stat:8.2f}  {p_value:12.4e}  {sig:>5s}  {ratio:6.2f}x")

    lines.append("")
    lines.append("=" * 70)
    lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")
    lines.append("Ratio: Top-K embedding change / Random-K embedding change (higher = more faithful)")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    # Save to file
    report_path = os.path.join(save_dir, 'perturbation_statistics.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n  Report saved: {report_path}")

    return report


# =====================================================================
# MAIN
# =====================================================================

def main():
    print()
    print("=" * 70)
    print("VALIDATION: PERTURBATION FAITHFULNESS TEST (LL-1 & LL-2)")
    print("=" * 70)
    print()

    # --- Configuration ---
    DATASET = "kiba"
    DRUG_K_VALUES = [3, 5, 10]
    PROTEIN_K_VALUES = [5, 10, 20]
    N_RANDOM_TRIALS = 10
    SAVE_DIR = os.path.join('results', 'validation', 'perturbation')
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device('cpu')
    print(f"  Dataset:        {DATASET}")
    print(f"  Drug K values:  {DRUG_K_VALUES}")
    print(f"  Protein K vals: {PROTEIN_K_VALUES}")
    print(f"  Random trials:  {N_RANDOM_TRIALS}")
    print(f"  Save Dir:       {SAVE_DIR}")
    print()

    # --- Step 1: Load saved explainability results ---
    print("Step 1: Loading saved explainability results...")
    ll_results_path = os.path.join('results', 'explainability_lowlevel', 'all_results.pt')
    ll_results = torch.load(ll_results_path, map_location='cpu')
    print(f"  Loaded {len(ll_results)} low-level results")

    # --- Step 2: Load model and data checkpoints ---
    print("Step 2: Loading checkpoints...")
    data_ckpt = torch.load(f'checkpoints/data_{DATASET}.pt', map_location=device)
    data_new = data_ckpt['data_new']
    nb_drugs = data_ckpt['nb_drugs']
    nb_proteins = data_ckpt['nb_proteins']

    args = opt.parser.parse_args([])
    gnnnet = GNNNet().to(device)
    gnnnet.load_state_dict(torch.load(f'checkpoints/gnnnet_{DATASET}.pt', map_location=device))
    gnnnet.eval()
    print(f"  Loaded GNNNet from checkpoints/gnnnet_{DATASET}.pt")
    print()

    # --- Step 3: Run perturbation tests ---
    print("Step 3: Running perturbation tests...")
    print()

    drugmap, proteinmap, _, _ = build_drug_protein_maps(data_new)

    all_drug_perturb = []
    all_protein_perturb = []
    
    np.random.seed(42)

    for i, result in enumerate(ll_results):
        drug_id = result['drug_id']
        protein_id = result['protein_id']
        drug_imp = result['drug_node_importance']
        pro_imp = result['protein_node_importance']
        gt = result['ground_truth']

        if isinstance(drug_imp, torch.Tensor):
            drug_imp = drug_imp.numpy()
        if isinstance(pro_imp, torch.Tensor):
            pro_imp = pro_imp.numpy()

        print(f"  Sample {i:3d}/{len(ll_results)}: Drug={drug_id}, Protein={protein_id}, GT={gt}")

        # Drug perturbation
        try:
            drug_data, smiles = get_drug_graph_data(drug_id, data_new, drugmap)
            drug_data = drug_data.to(device)
            drug_perturb = run_perturbation_test_drug(
                gnnnet, drug_data, drug_imp, device,
                K_values=DRUG_K_VALUES, n_random_trials=N_RANDOM_TRIALS
            )
            all_drug_perturb.append(drug_perturb)

            K_mid = DRUG_K_VALUES[len(DRUG_K_VALUES) // 2]
            if K_mid in drug_perturb:
                print(f"    Drug  K={K_mid}: top={drug_perturb[K_mid]['top_k_drop']:.6f}, "
                      f"rand={drug_perturb[K_mid]['random_k_mean']:.6f}")
        except Exception as e:
            print(f"    [ERROR] Drug perturbation failed: {e}")
            all_drug_perturb.append({})

        # Protein perturbation
        try:
            pro_data, sequence = get_protein_graph_data(protein_id, data_new, DATASET)
            pro_data = pro_data.to(device)
            protein_perturb = run_perturbation_test_protein(
                gnnnet, pro_data, pro_imp, device,
                K_values=PROTEIN_K_VALUES, n_random_trials=N_RANDOM_TRIALS
            )
            all_protein_perturb.append(protein_perturb)

            K_mid = PROTEIN_K_VALUES[len(PROTEIN_K_VALUES) // 2]
            if K_mid in protein_perturb:
                print(f"    Prot  K={K_mid}: top={protein_perturb[K_mid]['top_k_drop']:.6f}, "
                      f"rand={protein_perturb[K_mid]['random_k_mean']:.6f}")
        except Exception as e:
            print(f"    [ERROR] Protein perturbation failed: {e}")
            all_protein_perturb.append({})

    # --- Step 4: Filter out empty results ---
    all_drug_perturb = [r for r in all_drug_perturb if r]
    all_protein_perturb = [r for r in all_protein_perturb if r]
    print(f"\n  Valid drug results:    {len(all_drug_perturb)}")
    print(f"  Valid protein results: {len(all_protein_perturb)}")

    # --- Step 5: Statistical summary ---
    print("\nStep 4: Statistical analysis...")
    print_statistical_summary(
        all_drug_perturb, all_protein_perturb,
        DRUG_K_VALUES, PROTEIN_K_VALUES, SAVE_DIR
    )

    # --- Step 6: Visualization ---
    print("\nStep 5: Generating visualizations...")
    plot_perturbation_results(
        all_drug_perturb, all_protein_perturb,
        DRUG_K_VALUES, PROTEIN_K_VALUES, SAVE_DIR
    )

    # --- Step 7: Save raw results ---
    raw_path = os.path.join(SAVE_DIR, 'perturbation_results.pt')
    torch.save({
        'drug_results': all_drug_perturb,
        'protein_results': all_protein_perturb,
        'drug_K_values': DRUG_K_VALUES,
        'protein_K_values': PROTEIN_K_VALUES,
    }, raw_path)
    print(f"  Raw results saved: {raw_path}")

    print()
    print("=" * 70)
    print("PERTURBATION FAITHFULNESS TEST COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
