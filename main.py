from opt import *
from utils import *
from model import H2GNN
import time
import random
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
from data_load import dataload
from setting import process
from NodeRepresentation import GNNNet,combined
import torch
import matplotlib.pyplot as plt

args = parser.parse_args()
start_time = time.time()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def normalize_features_torch(mx):
    """Row-normalize a dense feature matrix (differentiable)."""
    rowsum = mx.sum(dim=1, keepdim=True)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    return mx * r_inv

"""Load preprocessed data."""
# DATASET = "davis"
DATASET = "kiba"
# DATASET = "DrugBank"

data_new, nb_drugs, nb_proteins = dataload(DATASET)
nb_all = nb_drugs+nb_proteins
drug_set, protein_set, adj, labels, idx_train, idx_val, idx_test, edge = process(
    data_new, nb_drugs, nb_proteins, DATASET, foldcount=5, setting=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

drug_batch = None
pro_batch = None
for batch, (drug, pro) in enumerate(zip(drug_set, protein_set)):
    drug_batch, pro_batch = drug.to(device), pro.to(device)

node = GNNNet().to(device)
model = H2GNN(n_node=nb_all).to(device)
optimizer = optim.Adam(list(model.parameters()) + list(node.parameters()), lr=args.lr)
# adj_hat is already a probability (sigmoid in IGAE decoder); do not apply sigmoid again.
myloss = nn.BCELoss()
gamma_value = 0.3

adj = adj.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

acc_reuslt = []
f1_result = []

def encode_features():
    feats = node(drug_batch.x, drug_batch.edge_index, drug_batch.batch,
                 pro_batch.x, pro_batch.edge_index, pro_batch.batch)
    return normalize_features_torch(feats)

def pair_scores():
    """Forward H2GNN and return flattened drug-protein probabilities."""
    features = encode_features()
    x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde, alpha = model(features, adj)
    output = adj_hat[:nb_drugs, nb_drugs:nb_all]
    pre = output.reshape(-1)
    return pre, alpha, features

def Train(epoch):
    model.train()
    node.train()
    pre, alpha, _ = pair_scores()
    loss_train = myloss(pre[idx_train], labels[idx_train])

    # Gate Regularization: Encourages the model to use both AE and IGAE
    # instead of just picking one (alpha=0 or alpha=1)
    gate_penalty = -torch.mean(alpha * torch.log(alpha + 1e-10) + (1 - alpha) * torch.log(1 - alpha + 1e-10))

    total_loss = loss_train + (0.05 * gate_penalty)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

def evaluate(idx, threshold=None, split_name='val'):
    model.eval()
    node.eval()
    with torch.no_grad():
        pre, _, _ = pair_scores()
        loss = myloss(pre[idx], labels[idx])
        yp = pre[idx].cpu().detach().numpy()
        ytrue = labels[idx].cpu().detach().numpy()
        AUC, AUPR, F1, ACC, used_thr = metrics_graph(ytrue, yp, threshold=threshold)
        print(split_name + ' loss: ', str(round(loss.item(), 4)))
        print(split_name + ' auc: ' + str(round(AUC, 4)) + '  ' + split_name + ' aupr: ' + str(round(AUPR, 4)) +
              '  ' + split_name + ' f1: ' + str(round(F1, 4)) + '  ' + split_name + ' acc: ' + str(round(ACC, 4)))
    return AUC, AUPR, F1, ACC, used_thr

#------main: select checkpoint on validation, report test once
best_val_auc = -1.0
best_epoch = 0
best_threshold = 0.5
best_h2gnn = None
best_gnnnet = None
for epoch in range(args.epochs):
    if epoch % 10 == 0:
        print('\nepoch: ' + str(epoch))
    Train(epoch)
    val_AUC, val_AUPR, val_F1, val_ACC, val_thr = evaluate(idx_val, threshold=None, split_name='val')
    if epoch % 10 == 0:
        evaluate(idx_test, threshold=val_thr, split_name='test_monitor')
    if val_AUC > best_val_auc:
        best_val_auc = val_AUC
        best_epoch = epoch
        best_threshold = val_thr
        best_h2gnn = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_gnnnet = {k: v.detach().cpu().clone() for k, v in node.state_dict().items()}

model.load_state_dict(best_h2gnn)
node.load_state_dict(best_gnnnet)
print('\n--- test (val-selected epoch, frozen F1 threshold) ---')
final_AUC, final_AUPR, final_F1, final_ACC, _ = evaluate(
    idx_test, threshold=best_threshold, split_name='test')

elapsed = time.time() - start_time
print('---------------------------------------')
print("Train in " + DATASET)
print('Elapsed time: ', round(elapsed, 4))
print("best_epoch (by val AUC): " + str(best_epoch))
print("val_AUC at best epoch: " + str(round(best_val_auc, 4)))
print("F1 threshold (from val): " + str(round(best_threshold, 6)))
print('Final_AUC: ' + str(round(final_AUC, 4)) + '  Final_AUPR: ' + str(round(final_AUPR, 4)) +
      '  Final_F1: ' + str(round(final_F1, 4)) + '  Final_ACC: ' + str(round(final_ACC, 4)))
print('---------------------------------------')

# === Save checkpoints for explainability ===
import os
os.makedirs('checkpoints', exist_ok=True)

torch.save(node.state_dict(), f'checkpoints/gnnnet_{DATASET}.pt')
torch.save(model.state_dict(), f'checkpoints/h2gnn_{DATASET}.pt')

node.eval()
model.eval()
with torch.no_grad():
    features = encode_features().detach().cpu()

# Save graph data and metadata needed for explainability
torch.save({
    'data_new': data_new,
    'nb_drugs': nb_drugs,
    'nb_proteins': nb_proteins,
    'nb_all': nb_all,
    'features': features,
    'adj': adj.cpu(),
    'labels': labels.cpu(),
    'idx_train': idx_train.cpu(),
    'idx_val': idx_val.cpu(),
    'idx_test': idx_test.cpu(),
    'edge': edge,
    'dataset': DATASET,
    'best_epoch': best_epoch,
    'f1_threshold': best_threshold,
}, f'checkpoints/data_{DATASET}.pt')

print(f'Checkpoints saved to checkpoints/ for dataset: {DATASET}')

def visualize_alpha(model, features, adj, nb_drugs, nb_all):
    model.eval()
    with torch.no_grad():
        # Get alpha from the updated forward pass
        _, _, _, _, _, _, alpha = model(features, adj)

    # Average across the latent dimension (n_z) to get 1 value per node
    # alpha_mean near 1 = Model prefers AE (Structure)
    # alpha_mean near 0 = Model prefers IGAE (Network)
    alpha_mean = torch.mean(alpha, dim=1).cpu().numpy()

    drug_alpha = alpha_mean[:nb_drugs]
    prot_alpha = alpha_mean[nb_drugs:nb_all]

    plt.figure(figsize=(10, 6))
    plt.hist(drug_alpha, bins=30, alpha=0.5, label='Drugs (AE Influence)', color='blue')
    plt.hist(prot_alpha, bins=30, alpha=0.5, label='Proteins (AE Influence)', color='green')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Neutral (Original H2GnnDTI)')

    plt.title('Distribution of Dynamic Weight (Alpha) across Nodes')
    plt.xlabel('Alpha Value (Higher = More Weight on Chemical Structure)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
