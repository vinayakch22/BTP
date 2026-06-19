import torch, numpy as np
from collections import Counter

# ============================================================
# LOW-LEVEL RESULTS
# ============================================================
ll = torch.load('results/explainability_lowlevel/all_results.pt', map_location='cpu')
print('=== LOW-LEVEL RESULTS ===')
print(f'Samples: {len(ll)}')
print(f'Keys: {list(ll[0].keys())}')

# First result detail
r0 = ll[0]
print(f"\nSample 0 detail:")
print(f"  drug_id: {r0['drug_id']}")
print(f"  protein_id: {r0['protein_id']}")
print(f"  ground_truth: {r0['ground_truth']}")
print(f"  prediction: {r0['prediction']}")
print(f"  drug_node_importance type: {type(r0['drug_node_importance'])}")
print(f"  drug_top_atoms: {r0['drug_top_atoms']}")
print(f"  protein_top_residues: {r0['protein_top_residues']}")

if isinstance(r0['drug_node_importance'], torch.Tensor):
    print(f"  drug_node_importance shape: {r0['drug_node_importance'].shape}")
elif isinstance(r0['drug_node_importance'], np.ndarray):
    print(f"  drug_node_importance shape: {r0['drug_node_importance'].shape}")
elif isinstance(r0['drug_node_importance'], list):
    print(f"  drug_node_importance len: {len(r0['drug_node_importance'])}")

# Drug importance stats
drug_maxes = []
prot_maxes = []
for r in ll:
    di = r['drug_node_importance']
    pi = r['protein_node_importance']
    if isinstance(di, torch.Tensor):
        drug_maxes.append(di.float().max().item())
    elif isinstance(di, np.ndarray):
        drug_maxes.append(float(np.max(di)))
    elif isinstance(di, list):
        drug_maxes.append(float(max(di)))
    
    if isinstance(pi, torch.Tensor):
        prot_maxes.append(pi.float().max().item())
    elif isinstance(pi, np.ndarray):
        prot_maxes.append(float(np.max(pi)))
    elif isinstance(pi, list):
        prot_maxes.append(float(max(pi)))

print(f"\nDrug node importance max: mean={np.mean(drug_maxes):.8f}, std={np.std(drug_maxes):.8f}")
print(f"Protein node importance max: mean={np.mean(prot_maxes):.8f}, std={np.std(prot_maxes):.8f}")

# Top atom frequency
atom_freq = Counter()
for r in ll:
    for a in r['drug_top_atoms'][:15]:
        atom_freq[a] += 1
print(f"\nTop atom indices (most frequent):")
for idx, count in atom_freq.most_common(10):
    print(f"  Atom {idx}: {count}/100 samples")

# Top residue frequency
res_freq = Counter()
for r in ll:
    for a in r['protein_top_residues'][:15]:
        res_freq[a] += 1
print(f"\nTop residue indices (most frequent):")
for idx, count in res_freq.most_common(10):
    print(f"  Residue {idx}: {count}/100 samples")

# Predictions distribution
preds = [r['prediction'] for r in ll]
gts = [r['ground_truth'] for r in ll]
print(f"\nPredictions: mean={np.mean(preds):.4f}, std={np.std(preds):.4f}, min={np.min(preds):.4f}, max={np.max(preds):.4f}")
print(f"Ground truths: unique={set(gts)}, count_0={gts.count(0)}, count_1={gts.count(1) if 1 in gts else 'N/A'}")

# ============================================================
# PREDICTION-LEVEL RESULTS
# ============================================================
pl = torch.load('results/explainability_prediction/all_results.pt', map_location='cpu')
print(f'\n=== PREDICTION-LEVEL RESULTS ===')
print(f'Samples: {len(pl)}')
print(f'Keys: {list(pl[0].keys())}')

# First result detail
p0 = pl[0]
print(f"\nSample 0 detail:")
for k, v in p0.items():
    if isinstance(v, torch.Tensor):
        if v.numel() < 10:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: shape={v.shape}, mean={v.float().mean():.6f}")
    elif isinstance(v, np.ndarray):
        if v.size < 10:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: shape={v.shape}, mean={v.mean():.6f}")
    elif isinstance(v, (list, tuple)):
        print(f"  {k}: len={len(v)}, first5={v[:5]}")
    else:
        print(f"  {k}: {v}")

# Alpha gate analysis
alpha_means = []
ae_fracs = []
igae_fracs = []
for r in pl:
    if 'alpha_mean' in r:
        v = r['alpha_mean']
        alpha_means.append(v.item() if isinstance(v, torch.Tensor) else float(v))
    if 'ae_fraction' in r:
        v = r['ae_fraction']
        ae_fracs.append(v.item() if isinstance(v, torch.Tensor) else float(v))
    if 'igae_fraction' in r:
        v = r['igae_fraction']
        igae_fracs.append(v.item() if isinstance(v, torch.Tensor) else float(v))

if alpha_means:
    print(f"\nAlpha gate stats:")
    print(f"  Mean: {np.mean(alpha_means):.6f}")
    print(f"  Std:  {np.std(alpha_means):.6f}")
    print(f"  Min:  {np.min(alpha_means):.6f}")
    print(f"  Max:  {np.max(alpha_means):.6f}")

if ae_fracs:
    print(f"\nAE fraction: mean={np.mean(ae_fracs):.6f}, std={np.std(ae_fracs):.6f}")
    print(f"IGAE fraction: mean={np.mean(igae_fracs):.6f}, std={np.std(igae_fracs):.6f}")
    ae_dominant = sum(1 for a in ae_fracs if a > 0.5)
    igae_dominant = sum(1 for a in igae_fracs if a > 0.5)
    print(f"AE dominant (>50%): {ae_dominant}/100")
    print(f"IGAE dominant (>50%): {igae_dominant}/100")

# Predictions
preds_pl = [r['prediction_raw'] if 'prediction_raw' in r else r.get('prediction', 0) for r in pl]
gts_pl = [r['ground_truth'] for r in pl]
print(f"\nPrediction stats:")
print(f"  Mean: {np.mean(preds_pl):.4f}, Std: {np.std(preds_pl):.4f}")
print(f"  Ground truth: pos={sum(1 for g in gts_pl if g>0)}, neg={sum(1 for g in gts_pl if g==0)}")

# Feature attributions
if 'attr_features_drug' in pl[0]:
    drug_attr = [np.abs(r['attr_features_drug'].numpy() if isinstance(r['attr_features_drug'], torch.Tensor) else r['attr_features_drug']).mean() for r in pl]
    prot_attr = [np.abs(r['attr_features_protein'].numpy() if isinstance(r['attr_features_protein'], torch.Tensor) else r['attr_features_protein']).mean() for r in pl]
    print(f"\nFeature attribution (abs mean):")
    print(f"  Drug: {np.mean(drug_attr):.8f}")
    print(f"  Protein: {np.mean(prot_attr):.8f}")
    print(f"  Drug/Protein ratio: {np.mean(drug_attr)/np.mean(prot_attr):.4f}")

if 'nonzero_drug' in pl[0]:
    nz_drug = [r['nonzero_drug'] for r in pl]
    nz_prot = [r['nonzero_protein'] for r in pl]
    print(f"\nNon-zero attributions:")
    print(f"  Drug: mean={np.mean(nz_drug):.1f}, std={np.std(nz_drug):.1f}")
    print(f"  Protein: mean={np.mean(nz_prot):.1f}, std={np.std(nz_prot):.1f}")

print("\nDONE!")
