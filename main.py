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

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

"""Load preprocessed data."""
# DATASET = "davis"
DATASET = "kiba"
# DATASET = "DrugBank"

data_new, nb_drugs, nb_proteins = dataload(DATASET)
nb_all = nb_drugs+nb_proteins
drug_set, protein_set, adj, labels, idx_train, idx_test,edge = process(data_new, nb_drugs, nb_proteins,DATASET,foldcount=5,setting = 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

node = GNNNet().to(device)

for batch, (drug, pro) in enumerate(zip(drug_set, protein_set)):
    drug, pro = drug.to(device), pro.to(device)
    features = node(drug.x, drug.edge_index, drug.batch, pro.x, pro.edge_index, pro.batch)

features = normalize_features(features.detach().cpu().numpy())
features = torch.FloatTensor(features).to(device)

print(f"DEBUG: Features shape: {features.shape}") 
print(f"DEBUG: Expected n_input: {args.n_input}")

if features.shape[1] != args.n_input:
    print("CRITICAL ERROR: Feature dimension does not match model input!")

model = H2GNN(n_node=features.shape[0])
optimizer = optim.Adam(model.parameters(),lr=args.lr)
myloss = nn.BCEWithLogitsLoss()
gamma_value = 0.3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
features = features.to(device)
adj = adj.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_test = idx_test.to(device)

acc_reuslt = []
f1_result = []

def Train(epoch):
    model.train()
    # Unpack 7 values
    x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde, alpha = model(features, adj)
    
    output = adj_hat[:nb_drugs, nb_drugs:nb_all + 1]
    pre = output.reshape(-1)
    pre = torch.sigmoid(pre)
    
    loss_train = myloss(pre[idx_train], labels[idx_train])
    
    # Gate Regularization: Encourages the model to use both AE and IGAE 
    # instead of just picking one (alpha=0 or alpha=1)
    gate_penalty = -torch.mean(alpha * torch.log(alpha + 1e-10) + (1 - alpha) * torch.log(1 - alpha + 1e-10))
    
    total_loss = loss_train + (0.05 * gate_penalty)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

def test():
    model.eval()
    with torch.no_grad():
        # Add 'alpha' to the end of this list
        x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde, alpha = model(features, adj)
        
        output = adj_hat[:nb_drugs, nb_drugs:nb_all + 1]
        pre = output.reshape(-1)
        loss_test = myloss(pre[idx_test], labels[idx_test])  ##BCEloss
        # loss_ae = F.mse_loss(x_hat, features)
        loss = loss_test
        yp = pre[idx_test].cpu().detach().numpy()
        ytest = labels[idx_test].cpu().detach().numpy()
        AUC, AUPR, F1, ACC = metrics_graph(ytest,yp)
        print('test loss: ', str(round(loss.item(), 4)))
        print('test auc: ' + str(round(AUC, 4)) + '  test aupr: ' + str(round(AUPR, 4)) +
              '  test f1: ' + str(round(F1, 4)) + '  test acc: ' + str(round(ACC, 4)))
    return AUC, AUPR, F1, ACC

#------main
final_AUC = 0;final_AUPR = 0;final_F1 = 0;final_ACC = 0
for epoch in range(args.epochs):
    if epoch % 10 == 0:
        print('\nepoch: ' + str(epoch))
    Train(epoch)
    AUC, AUPR, F1, ACC = test()
    if (AUC > final_AUC):
        best_epoch = epoch
        final_AUC = AUC;final_AUPR = AUPR;final_F1 = F1;final_ACC = ACC
elapsed = time.time() - start_time
print('---------------------------------------')
print("Train in " + DATASET)
print('Elapsed time: ', round(elapsed, 4))
print("best_epoch: " + str(best_epoch))
print('Final_AUC: ' + str(round(final_AUC, 4)) + '  Final_AUPR: ' + str(round(final_AUPR, 4)) +
      '  Final_F1: ' + str(round(final_F1, 4)) + '  Final_ACC: ' + str(round(final_ACC, 4)))
print('---------------------------------------')

# === Save checkpoints for explainability ===
import os
os.makedirs('checkpoints', exist_ok=True)

torch.save(node.state_dict(), f'checkpoints/gnnnet_{DATASET}.pt')
torch.save(model.state_dict(), f'checkpoints/h2gnn_{DATASET}.pt')

# Save graph data and metadata needed for explainability
torch.save({
    'data_new': data_new,
    'nb_drugs': nb_drugs,
    'nb_proteins': nb_proteins,
    'nb_all': nb_all,
    'features': features.cpu(),
    'adj': adj.cpu(),
    'labels': labels.cpu(),
    'idx_train': idx_train.cpu(),
    'idx_test': idx_test.cpu(),
    'edge': edge,
    'dataset': DATASET,
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