"""
=================================================================
VALIDATION: PHARMACOPHORE OVERLAP (LL-4)
=================================================================

OBJECTIVE:
  Validate whether the drug atoms identified as "important" by
  Integrated Gradients correspond to known pharmacophoric features
  (chemically meaningful functional groups).

WHY THIS MATTERS:
  If IG highlights random atoms with no chemical significance,
  the explainability is unreliable. But if it highlights atoms
  that ARE part of known drug-activity-relevant functional groups
  (H-bond donors, acceptors, aromatic rings, charged groups),
  it confirms the model has learned chemically meaningful patterns.

METHODOLOGY:
  For each of the 100 drug molecules:
  1. Parse the SMILES string using RDKit.
  2. Identify all pharmacophoric atoms (H-bond donors, acceptors,
     aromatic atoms, positively/negatively charged atoms, and
     hydrophobic atoms) using RDKit's built-in pharmacophore
     feature factory.
  3. Get the top-K most important atoms from our IG results.
  4. Compute:
     - Overlap ratio = |Top-K intersect Pharmacophore| / K
     - Random baseline = |Random-K intersect Pharmacophore| / K
     - Enrichment = Overlap ratio / Random baseline
  5. Run across all 100 samples and perform statistical tests.
=================================================================
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# =====================================================================
# PHARMACOPHORE FEATURE IDENTIFICATION
# =====================================================================

def get_pharmacophore_atoms(mol):
    """
    Identify pharmacophoric atoms in a molecule using RDKit.
    Returns a dict of category -> set of atom indices.

    Categories:
      - hbd: Hydrogen bond donors (N-H, O-H)
      - hba: Hydrogen bond acceptors (N, O with lone pairs)
      - aromatic: Atoms in aromatic rings
      - pos_charged: Positively ionizable atoms
      - neg_charged: Negatively ionizable atoms
      - hydrophobic: Hydrophobic atoms (non-polar carbon)
    """
    if mol is None:
        return {}, set()

    num_atoms = mol.GetNumAtoms()

    hbd = set()  # H-bond donors
    hba = set()  # H-bond acceptors
    aromatic = set()  # Aromatic atoms
    pos_charged = set()  # Positively charged
    neg_charged = set()  # Negatively charged
    hydrophobic = set()  # Hydrophobic

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        is_aromatic = atom.GetIsAromatic()
        num_hs = atom.GetTotalNumHs()
        formal_charge = atom.GetFormalCharge()

        # Aromatic atoms
        if is_aromatic:
            aromatic.add(idx)

        # H-bond donors: N or O with at least one H
        if symbol in ('N', 'O') and num_hs > 0:
            hbd.add(idx)

        # H-bond acceptors: N or O (have lone pairs)
        if symbol in ('N', 'O', 'F'):
            hba.add(idx)

        # Positively charged
        if formal_charge > 0:
            pos_charged.add(idx)

        # Negatively charged
        if formal_charge < 0:
            neg_charged.add(idx)

        # Hydrophobic: non-polar carbon (not bonded to N, O, or charged)
        if symbol == 'C' and not is_aromatic:
            neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
            if not any(n in ('N', 'O', 'S', 'F', 'Cl', 'Br', 'I') for n in neighbors):
                hydrophobic.add(idx)

    # Also identify heteroatoms (non-C, non-H) as generally pharmacophoric
    heteroatoms = set()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ('C', 'H'):
            heteroatoms.add(atom.GetIdx())

    categories = {
        'hbd': hbd,
        'hba': hba,
        'aromatic': aromatic,
        'pos_charged': pos_charged,
        'neg_charged': neg_charged,
        'hydrophobic': hydrophobic,
        'heteroatom': heteroatoms,
    }

    # Union of all pharmacophoric atoms
    all_pharma = set()
    for cat_set in categories.values():
        all_pharma.update(cat_set)

    return categories, all_pharma


# =====================================================================
# OVERLAP COMPUTATION
# =====================================================================

def compute_overlap_stats(top_k_atoms, pharma_atoms, all_atom_count, K, n_random_trials=100):
    """
    Compute overlap between top-K attributed atoms and pharmacophoric atoms.
    Also compute random baseline.
    """
    # Overlap of top-K with pharmacophore
    top_k_set = set(top_k_atoms[:K])
    overlap = top_k_set & pharma_atoms
    overlap_ratio = len(overlap) / K if K > 0 else 0.0

    # Random baseline: average overlap of random K atoms
    random_overlaps = []
    for trial in range(n_random_trials):
        random_k = set(np.random.choice(all_atom_count, min(K, all_atom_count), replace=False))
        random_overlap = random_k & pharma_atoms
        random_overlaps.append(len(random_overlap) / K if K > 0 else 0.0)

    random_mean = np.mean(random_overlaps)
    enrichment = overlap_ratio / random_mean if random_mean > 0 else float('inf')

    return overlap_ratio, random_mean, enrichment


# =====================================================================
# MAIN VALIDATION
# =====================================================================

def main():
    print()
    print("=" * 70)
    print("VALIDATION: PHARMACOPHORE OVERLAP (LL-4)")
    print("=" * 70)
    print()

    SAVE_DIR = os.path.join('results', 'validation', 'pharmacophore')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Load saved results ---
    print("Loading saved low-level results...")
    ll_results = torch.load(
        os.path.join('results', 'explainability_lowlevel', 'all_results.pt'),
        map_location='cpu'
    )
    print(f"  Loaded {len(ll_results)} samples")

    # --- Load data_new to get SMILES ---
    print("Loading data checkpoint...")
    data_ckpt = torch.load('checkpoints/data_kiba.pt', map_location='cpu')
    data_new = data_ckpt['data_new']
    print(f"  Loaded data_new with {len(data_new)} entries")
    print()

    # Build drug_id -> SMILES mapping
    drug_smiles_map = {}
    for item in data_new:
        drug_id = item[0]
        smiles = item[3]
        if drug_id not in drug_smiles_map:
            drug_smiles_map[drug_id] = smiles

    np.random.seed(42)

    K_values = [3, 5, 10]

    # Per-sample results
    sample_results = []

    # Category-level tracking
    category_names = ['hbd', 'hba', 'aromatic', 'hydrophobic', 'heteroatom']
    category_hit_rates = {cat: {K: [] for K in K_values} for cat in category_names}

    print("Running pharmacophore overlap analysis...")
    print()

    for s_idx, result in enumerate(ll_results):
        drug_id = result['drug_id']
        imp = result['drug_node_importance']
        if isinstance(imp, torch.Tensor):
            imp = imp.numpy()

        # Get SMILES and parse molecule
        smiles = drug_smiles_map.get(drug_id)
        if smiles is None:
            print(f"  Sample {s_idx}: Drug {drug_id} - SMILES not found, skipping")
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  Sample {s_idx}: Drug {drug_id} - Invalid SMILES, skipping")
            continue

        num_atoms = mol.GetNumAtoms()
        categories, all_pharma = get_pharmacophore_atoms(mol)

        # Sort atoms by importance (descending)
        sorted_atoms = np.argsort(imp[:num_atoms])[::-1]

        pharma_fraction = len(all_pharma) / num_atoms if num_atoms > 0 else 0

        sample_data = {
            'drug_id': drug_id,
            'num_atoms': num_atoms,
            'num_pharma': len(all_pharma),
            'pharma_fraction': pharma_fraction,
            'ground_truth': result['ground_truth'],
        }

        for K in K_values:
            actual_k = min(K, num_atoms)
            overlap_ratio, random_mean, enrichment = compute_overlap_stats(
                sorted_atoms.tolist(), all_pharma, num_atoms, actual_k
            )
            sample_data[f'overlap_K{K}'] = overlap_ratio
            sample_data[f'random_K{K}'] = random_mean
            sample_data[f'enrichment_K{K}'] = enrichment

            # Per-category hit rate
            top_k_set = set(sorted_atoms[:actual_k].tolist())
            for cat_name in category_names:
                cat_atoms = categories.get(cat_name, set())
                if len(cat_atoms) > 0:
                    hits = len(top_k_set & cat_atoms)
                    category_hit_rates[cat_name][K].append(hits / actual_k)

        sample_results.append(sample_data)

        if s_idx < 5 or s_idx % 20 == 0:
            print(f"  Sample {s_idx:3d}: Drug={drug_id}, Atoms={num_atoms}, "
                  f"Pharma={len(all_pharma)} ({100*pharma_fraction:.0f}%), "
                  f"Top-5 overlap={sample_data['overlap_K5']:.2f}, "
                  f"Enrichment={sample_data['enrichment_K5']:.2f}x")

    print()
    print(f"  Successfully analyzed {len(sample_results)} samples")
    print()

    # ===================================================================
    # STATISTICAL ANALYSIS
    # ===================================================================
    print("=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    print()

    # --- Overall statistics ---
    pharma_fractions = [s['pharma_fraction'] for s in sample_results]
    print(f"  Average pharmacophore fraction: {np.mean(pharma_fractions):.3f} "
          f"(i.e., {100*np.mean(pharma_fractions):.1f}% of atoms are pharmacophoric)")
    print()

    # --- Overlap vs Random for each K ---
    print("-" * 70)
    print(f"{'K':>5s}  {'Top-K Overlap':>14s}  {'Random Overlap':>14s}  "
          f"{'Enrichment':>11s}  {'Wilcoxon p':>12s}  {'Sig':>5s}")
    print("-" * 70)

    test_results = {}
    for K in K_values:
        overlaps = [s[f'overlap_K{K}'] for s in sample_results]
        randoms = [s[f'random_K{K}'] for s in sample_results]
        enrichments = [s[f'enrichment_K{K}'] for s in sample_results]

        # Wilcoxon signed-rank test (paired: overlap vs random)
        try:
            w_stat, w_p = wilcoxon(overlaps, randoms, alternative='greater')
        except Exception:
            w_stat, w_p = 0, 1.0

        sig = '***' if w_p < 0.001 else '**' if w_p < 0.01 else '*' if w_p < 0.05 else 'n.s.'

        print(f"{K:5d}  {np.mean(overlaps):14.4f}  {np.mean(randoms):14.4f}  "
              f"{np.mean(enrichments):11.2f}x  {w_p:12.4e}  {sig:>5s}")

        test_results[K] = {
            'overlap_mean': np.mean(overlaps),
            'random_mean': np.mean(randoms),
            'enrichment_mean': np.mean(enrichments),
            'wilcoxon_p': w_p,
        }

    print("-" * 70)
    print()

    # --- Per-category hit rate ---
    print("PER-CATEGORY HIT RATE (K=5):")
    print("-" * 50)
    print(f"{'Category':<20s}  {'Hit Rate':>10s}  {'Samples':>8s}")
    print("-" * 50)
    for cat_name in category_names:
        rates = category_hit_rates[cat_name][5]
        if rates:
            print(f"{cat_name:<20s}  {np.mean(rates):10.4f}  {len(rates):8d}")
        else:
            print(f"{cat_name:<20s}  {'N/A':>10s}  {0:8d}")
    print("-" * 50)
    print()

    # --- Positive vs Negative comparison ---
    pos_enrichments = [s['enrichment_K5'] for s in sample_results if s['ground_truth'] == 1]
    neg_enrichments = [s['enrichment_K5'] for s in sample_results if s['ground_truth'] == 0]
    if pos_enrichments and neg_enrichments:
        u_stat, u_p = mannwhitneyu(pos_enrichments, neg_enrichments, alternative='two-sided')
        print(f"  Positive vs Negative enrichment (K=5):")
        print(f"    Positive: mean={np.mean(pos_enrichments):.3f}")
        print(f"    Negative: mean={np.mean(neg_enrichments):.3f}")
        print(f"    Mann-Whitney p={u_p:.4e}")
    print()

    # ===================================================================
    # VISUALIZATIONS
    # ===================================================================
    print("Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Plot 1: Overlap vs Random bar chart ---
    overlap_means = [test_results[K]['overlap_mean'] for K in K_values]
    random_means = [test_results[K]['random_mean'] for K in K_values]
    enrichments_mean = [test_results[K]['enrichment_mean'] for K in K_values]

    x = np.arange(len(K_values))
    width = 0.35
    bars1 = axes[0, 0].bar(x - width/2, overlap_means, width, label='Top-K (IG)', color='steelblue', alpha=0.8)
    bars2 = axes[0, 0].bar(x + width/2, random_means, width, label='Random-K', color='gray', alpha=0.6)
    axes[0, 0].set_xlabel('K (Number of Top Atoms)')
    axes[0, 0].set_ylabel('Fraction Overlapping with Pharmacophore')
    axes[0, 0].set_title('LL-4: Top-K IG Atoms vs Random Atoms\nPharmacophore Overlap')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'K={K}' for K in K_values])
    axes[0, 0].legend()

    # Add enrichment labels
    for i, (K, enr) in enumerate(zip(K_values, enrichments_mean)):
        p_val = test_results[K]['wilcoxon_p']
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        axes[0, 0].text(i, max(overlap_means[i], random_means[i]) + 0.02,
                        f'{enr:.2f}x {sig}', ha='center', fontsize=10, fontweight='bold')

    axes[0, 0].set_ylim(0, max(overlap_means + random_means) * 1.3)

    # --- Plot 2: Enrichment distribution ---
    enr_data = [s['enrichment_K5'] for s in sample_results]
    axes[0, 1].hist(enr_data, bins=25, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='No enrichment (1.0x)')
    axes[0, 1].axvline(x=np.mean(enr_data), color='green', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(enr_data):.2f}x')
    axes[0, 1].set_xlabel('Enrichment (Top-K / Random-K)')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].set_title('LL-4: Pharmacophore Enrichment Distribution (K=5)')
    axes[0, 1].legend()

    # --- Plot 3: Per-category hit rate ---
    cat_means = []
    cat_labels = []
    for cat_name in category_names:
        rates = category_hit_rates[cat_name][5]
        if rates:
            cat_means.append(np.mean(rates))
            cat_labels.append(cat_name.replace('_', ' ').title())
        else:
            cat_means.append(0)
            cat_labels.append(cat_name.replace('_', ' ').title())

    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
    bars = axes[1, 0].bar(range(len(cat_labels)), cat_means, color=colors, alpha=0.8)
    axes[1, 0].set_xlabel('Pharmacophore Category')
    axes[1, 0].set_ylabel('Mean Hit Rate in Top-5')
    axes[1, 0].set_title('LL-4: Which Pharmacophore Types Are Highlighted?')
    axes[1, 0].set_xticks(range(len(cat_labels)))
    axes[1, 0].set_xticklabels(cat_labels, rotation=30, ha='right')
    for bar, val in zip(bars, cat_means):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', fontsize=9)

    # --- Plot 4: Scatter — Pharmacophore fraction vs Enrichment ---
    pf = [s['pharma_fraction'] for s in sample_results]
    enr5 = [s['enrichment_K5'] for s in sample_results]
    gt_colors = ['green' if s['ground_truth'] == 1 else 'red' for s in sample_results]
    axes[1, 1].scatter(pf, enr5, c=gt_colors, alpha=0.6, s=40)
    axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No enrichment')
    axes[1, 1].set_xlabel('Pharmacophore Fraction (in molecule)')
    axes[1, 1].set_ylabel('Enrichment (K=5)')
    axes[1, 1].set_title('Pharmacophore Density vs Enrichment')
    axes[1, 1].legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Positive'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Negative'),
        plt.Line2D([0], [0], color='gray', linestyle='--', label='No enrichment'),
    ])

    plt.suptitle('LL-4: Pharmacophore Overlap Validation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, 'pharmacophore_overlap.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # --- Additional: Example molecule visualization (first 5) ---
    fig2, axes2 = plt.subplots(1, 5, figsize=(25, 5))
    for plot_idx in range(min(5, len(sample_results))):
        s = sample_results[plot_idx]
        drug_id = s['drug_id']
        smiles = drug_smiles_map.get(drug_id)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        result = ll_results[plot_idx]
        imp = result['drug_node_importance']
        if isinstance(imp, torch.Tensor):
            imp = imp.numpy()

        num_atoms = mol.GetNumAtoms()
        categories, all_pharma = get_pharmacophore_atoms(mol)

        sorted_atoms = np.argsort(imp[:num_atoms])[::-1]
        top5 = set(sorted_atoms[:5].tolist())

        # Color atoms: green=top5+pharma, blue=top5 only, orange=pharma only, gray=neither
        atom_colors = {}
        for a_idx in range(num_atoms):
            in_top = a_idx in top5
            in_pharma = a_idx in all_pharma
            if in_top and in_pharma:
                atom_colors[a_idx] = (0.2, 0.8, 0.2)  # Green: overlap
            elif in_top:
                atom_colors[a_idx] = (0.2, 0.4, 0.9)  # Blue: IG only
            elif in_pharma:
                atom_colors[a_idx] = (1.0, 0.6, 0.1)  # Orange: pharma only
            else:
                atom_colors[a_idx] = (0.85, 0.85, 0.85)  # Gray

        img = Draw.MolToImage(mol, size=(300, 300),
                              highlightAtoms=list(range(num_atoms)),
                              highlightAtomColors=atom_colors)
        axes2[plot_idx].imshow(img)
        overlap_count = len(top5 & all_pharma)
        axes2[plot_idx].set_title(f'{drug_id}\nOverlap: {overlap_count}/5', fontsize=9)
        axes2[plot_idx].axis('off')

    plt.suptitle('LL-4: Example Molecules (Green=IG+Pharma overlap, Blue=IG only, Orange=Pharma only)',
                 fontsize=12)
    plt.tight_layout()
    fig2_path = os.path.join(SAVE_DIR, 'pharmacophore_examples.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig2_path}")

    # ===================================================================
    # SAVE RESULTS
    # ===================================================================
    torch.save({
        'sample_results': sample_results,
        'test_results': test_results,
        'category_hit_rates': {cat: {K: vals for K, vals in kv.items()}
                               for cat, kv in category_hit_rates.items()},
    }, os.path.join(SAVE_DIR, 'pharmacophore_results.pt'))

    # --- Write report ---
    lines = []
    lines.append("=" * 70)
    lines.append("LL-4: PHARMACOPHORE OVERLAP VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Samples analyzed: {len(sample_results)}")
    lines.append(f"Avg pharmacophore fraction: {np.mean(pharma_fractions):.3f} "
                 f"({100*np.mean(pharma_fractions):.1f}% of atoms)")
    lines.append("")
    lines.append("OVERLAP vs RANDOM:")
    lines.append("-" * 50)
    for K in K_values:
        r = test_results[K]
        sig = '***' if r['wilcoxon_p'] < 0.001 else '**' if r['wilcoxon_p'] < 0.01 else '*' if r['wilcoxon_p'] < 0.05 else 'n.s.'
        lines.append(f"  K={K}: Top-K={r['overlap_mean']:.4f}, "
                     f"Random={r['random_mean']:.4f}, "
                     f"Enrichment={r['enrichment_mean']:.2f}x, "
                     f"p={r['wilcoxon_p']:.4e} [{sig}]")
    lines.append("")
    lines.append("PER-CATEGORY HIT RATE (K=5):")
    lines.append("-" * 50)
    for cat_name in category_names:
        rates = category_hit_rates[cat_name][5]
        if rates:
            lines.append(f"  {cat_name:<20s}: {np.mean(rates):.4f}")
    lines.append("")
    if pos_enrichments and neg_enrichments:
        lines.append("POSITIVE vs NEGATIVE ENRICHMENT (K=5):")
        lines.append(f"  Positive: {np.mean(pos_enrichments):.3f}")
        lines.append(f"  Negative: {np.mean(neg_enrichments):.3f}")
        lines.append(f"  Mann-Whitney p={u_p:.4e}")
    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    report_path = os.path.join(SAVE_DIR, 'pharmacophore_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    print()
    print("=" * 70)
    print("PHARMACOPHORE OVERLAP VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
