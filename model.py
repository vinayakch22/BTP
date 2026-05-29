import opt
import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch_geometric.nn import GATConv

# device = torch.device('cuda:0') # unused globally
# AE encoder from DFCN
class AE_encoder(nn.Module):
    """AutoEncoder Encoder for capturing structural representations from node features."""
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, n_input, n_z):
        super(AE_encoder, self).__init__()
        self.enc_1 = Linear(n_input, ae_n_enc_1)  ##(160,128)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)  ##(128,256)
        self.enc_3 = Linear(ae_n_enc_2, ae_n_enc_3)  ##(256,512)
        self.z_layer = Linear(ae_n_enc_3, n_z)  ##(512,20)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        z = self.act(self.enc_1(x))
        z = self.act(self.enc_2(z))
        z = self.act(self.enc_3(z))
        z_ae = self.z_layer(z)
        return z_ae


# AE decoder from DFCN
class AE_decoder(nn.Module):
    """AutoEncoder Decoder for reconstructing features from latent representation."""
    def __init__(self, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE_decoder, self).__init__()

        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.dec_3 = Linear(ae_n_dec_2, ae_n_dec_3)
        self.x_bar_layer = Linear(ae_n_dec_3, n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_ae):
        z = self.act(self.dec_1(z_ae))
        z = self.act(self.dec_2(z))
        z = self.act(self.dec_3(z))
        x_hat = self.x_bar_layer(z)
        return x_hat


# Auto Encoder from DFCN
class AE(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE, self).__init__()

        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            n_input=n_input,
            n_z=n_z)

        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)

# GNNLayer from DFCN
class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = nn.Tanh()
        self.dropout1 = nn.Dropout(0.1)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        if active:
            support = self.act(torch.mm(features, self.weight))
        else:
            support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        return output


# IGAE encoder from DFCN
class IGAE_encoder(nn.Module):
    """Improved Graph AutoEncoder Encoder for capturing topological graph information."""
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)  ###(256,20)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z_1 = self.gnn_1(x, adj, active=True)
        z_2 = self.gnn_2(z_1, adj, active=True)
        z_igae = self.gnn_3(z_2, adj, active=False)   ###(447,20)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj


# IGAE decoder from DFCN
class IGAE_decoder(nn.Module):
    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(gae_n_dec_1, gae_n_dec_2)   ###(20,256)
        self.gnn_5 = GNNLayer(gae_n_dec_2, gae_n_dec_3)   ###(256,128)
        self.gnn_6 = GNNLayer(gae_n_dec_3, n_input)   ###(128,160)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):  ###(447,20)
        z_1 = self.gnn_4(z_igae, adj, active=True)
        z_2 = self.gnn_5(z_1, adj, active=True)
        z_hat = self.gnn_6(z_2, adj, active=True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


# Improved Graph Auto Encoder from DFCN
class IGAE(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE, self).__init__()
        # IGAE encoder
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        # IGAE decoder
        self.decoder = IGAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)

# Dual Correlation Reduction Network
class H2GNN(nn.Module):
    def __init__(self, n_node=None,dropout = 0.3):
        super(H2GNN, self).__init__()
        # Auto Encoder
        self.ae = AE(
            ae_n_enc_1=opt.args.ae_n_enc_1,
            ae_n_enc_2=opt.args.ae_n_enc_2,
            ae_n_enc_3=opt.args.ae_n_enc_3,
            ae_n_dec_1=opt.args.ae_n_dec_1,
            ae_n_dec_2=opt.args.ae_n_dec_2,
            ae_n_dec_3=opt.args.ae_n_dec_3,
            n_input=opt.args.n_input,
            n_z=opt.args.n_z)

        # Improved Graph Auto Encoder From DFCN
        self.gae = IGAE(
            gae_n_enc_1=opt.args.gae_n_enc_1,
            gae_n_enc_2=opt.args.gae_n_enc_2,
            gae_n_enc_3=opt.args.gae_n_enc_3,
            gae_n_dec_1=opt.args.gae_n_dec_1,
            gae_n_dec_2=opt.args.gae_n_dec_2,
            gae_n_dec_3=opt.args.gae_n_dec_3,
            n_input=opt.args.n_input)
        
        self.gat_refine = GATConv(opt.args.n_z, opt.args.n_z, heads=4, concat=False, dropout=dropout)
        # self.linear = torch.nn.Linear(opt.args.n_input, 1) # unused
        self.dropout = dropout
        self.gamma = 0.7

        # NEW: Dynamic Gate instead of static self.a and self.b
        # Input: Concatenated embeddings (20 + 20 = 40)
        # Output: Weight alpha (20)
        self.fusion_gate = nn.Sequential(
            # Step 1: Project the 40-dim concatenated features into a deeper space
                nn.Linear(opt.args.n_z * 2, opt.args.n_z * 2),
                nn.BatchNorm1d(opt.args.n_z * 2), # Keeps training stable
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2), # Prevents the gate from over-relying on one expert
    
                # Step 2: Attention Layer
                # This layer looks for the "importance" of each feature
                nn.Linear(opt.args.n_z * 2, opt.args.n_z),
                nn.Sigmoid() # Produces the alpha weight (0 to 1)
        )

    def forward(self, x, adj):
        z_ae = self.ae.encoder(x)
        z_igae, z_igae_adj = self.gae.encoder(x, adj)
        # New logic for Dynamic Weighting:
        combined_z = torch.cat((z_ae, z_igae), dim=1)
        # Calculate the dynamic attention weight
        alpha = self.fusion_gate(combined_z) 

        # Apply the weighted fusion
        z_i = alpha * z_ae + (1 - alpha) * z_igae
        # new GAT refinement
        # Convert dense adjacency to sparse edge_index for PyG compatibility
        edge_index = adj.nonzero().t().contiguous()
        # Apply Attention mechanism to fused features
        z_i = F.elu(self.gat_refine(z_i, edge_index))
        z_l = torch.spmm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.gamma * z_g + z_l   ###(447,20)
        x_hat = self.ae.decoder(z_tilde)
        z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj)
        adj_hat = z_hat_adj

        return x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde, alpha





