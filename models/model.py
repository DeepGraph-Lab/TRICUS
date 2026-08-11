import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch_geometric.nn import GCNConv, GATConv, DenseGCNConv


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Linear(nn.Module):
    def __init__(self, rna_in, drug_in, out_size):
        super().__init__()
        self.linear_lnc = nn.Linear(rna_in, out_size)
        self.linear_mi = nn.Linear(rna_in, out_size)
        self.linear_drug = nn.Linear(drug_in, out_size)

    def forward(self, lnc_emb, mi_emb, drug_emb):
        new_lnc_emb = self.linear_lnc(lnc_emb)
        new_mi_emb = self.linear_mi(mi_emb)
        drug_emb = self.linear_drug(drug_emb)
        rna_emb = torch.concat([new_lnc_emb, new_mi_emb], dim=0)
        return rna_emb, drug_emb


class RNAEmbeddingAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RNAEmbeddingAdapter, self).__init__()
        hid_dim = in_dim // 2
        self.linear = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(in_dim, hid_dim)),
            ("gelu", QuickGELU()),
            ("fc2", nn.Linear(hid_dim, hid_dim // 2)),
            ("gelu", QuickGELU()),
            ("fc3", nn.Linear(hid_dim // 2, out_dim)),
        ]))

    def forward(self, x):
        x = self.linear(x)
        return x


class DrugEmbeddingAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DrugEmbeddingAdapter, self).__init__()
        hid_dim = in_dim // 2
        self.linear = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(in_dim, hid_dim)),
            ("gelu", QuickGELU()),
            ("fc2", nn.Linear(hid_dim, out_dim)),
        ]))

    def forward(self, x):
        x = self.linear(x)
        return x


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.input_dim = in_dim
        self.output_dim = out_dim

        self.conv = GCNConv(
            in_dim,
            out_dim,
            add_self_loops=False,
            normalize=False,
            bias=False
        )

        self.norm = nn.LayerNorm(out_dim)
        self.act = QuickGELU()

        with torch.no_grad():
            self.conv.lin.weight.uniform_(0.0, 1.0)

    def forward(self, feat, adj):

        row, col = adj.nonzero(as_tuple=True)
        edge_index = torch.stack([row, col], dim=0)
        edge_weight = adj[row, col]

        H = self.conv(feat, edge_index, edge_weight)
        H = self.act(H)
        H = self.norm(H)

        return H

class PyHeteroGATLayers(nn.Module):
    def __init__(self,in_dim, out_dim, num_heads=4, dropout=0.1):
        super(PyHeteroGATLayers, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        assert out_dim % num_heads == 0
        self.head_dim = out_dim // num_heads

        self.gat_rna = GATConv(
            (in_dim, in_dim),
            out_channels=self.head_dim,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=False,
            bias=False
        )
        self.gat_drug = GATConv(
            (in_dim, in_dim),
            out_channels=self.head_dim,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=False,
            bias=False
        )

        self.res_rna = nn.Linear(in_dim, out_dim, bias=False)
        self.res_drug = nn.Linear(in_dim, out_dim, bias=False)

        self.rna_norm = nn.LayerNorm(out_dim)
        self.drug_norm = nn.LayerNorm(out_dim)

        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, rna_feat, drug_feat, adj):
        num_rna, num_drug = adj.shape
        assert rna_feat.size(0) == num_rna
        assert drug_feat.size(0) == num_drug

        rna_idx, drug_idx = (adj>0).nonzero(as_tuple=True)
        # Drug -> RNA
        edge_index_drug_to_rna = torch.stack([drug_idx,rna_idx],dim=0)
        # RNA -> Drug
        edge_index_rna_to_drug = torch.stack([rna_idx,drug_idx],dim=0)

        rna_agg = self.gat_rna((drug_feat, rna_feat), edge_index_drug_to_rna)
        rna_agg = self.dropout(rna_agg)

        drug_agg = self.gat_drug((rna_feat, drug_feat), edge_index_rna_to_drug)  # [Nd, out_dim]
        drug_agg = self.dropout(drug_agg)

        rna_res = self.res_rna(rna_feat)  # [Nr, out_dim]
        drug_res = self.res_drug(drug_feat)  # [Nd, out_dim]

        rna_out = self.rna_norm(self.act(rna_agg) + rna_res)
        drug_out = self.drug_norm(self.act(drug_agg) + drug_res)

        return rna_out, drug_out


class Predictor(nn.Module):
    def __init__(self, in_dim, pred_dim):
        super(Predictor, self).__init__()
        self.rna_layer = nn.Linear(in_dim, pred_dim)
        self.drug_layer = nn.Linear(in_dim, pred_dim)

    def forward(self, p_feat, d_feat):
        new_p_feat = self.rna_layer(p_feat)
        new_d_feat = self.drug_layer(d_feat)
        return new_p_feat, new_d_feat

class MF(nn.Module):

    def __init__(self, n_rna, n_drug, rna_feat_dim, drug_feat_dim,
                 embedding_dim=128, dropout=0.0,num_prototypes=32):
        super(MF, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_prototypes = num_prototypes

        self.rna_feature_encoder = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(rna_feat_dim, embedding_dim * 2)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("fc2", nn.Linear(embedding_dim * 2, embedding_dim)),
        ]))

        self.drug_feature_encoder = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(drug_feat_dim, embedding_dim * 2)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("fc2", nn.Linear(embedding_dim * 2, embedding_dim)),
        ]))

        self.rna_prototypes = nn.Parameter(
            torch.randn(num_prototypes, embedding_dim)*0.01,
        )
        self.drug_prototypes = nn.Parameter(
            torch.randn(num_prototypes, embedding_dim) * 0.01,
        )

        self.rna_gate = nn.Linear(rna_feat_dim, num_prototypes)
        self.drug_gate = nn.Linear(drug_feat_dim, num_prototypes)

        self.alpha_rna = nn.Parameter(torch.tensor(0.5))
        self.alpha_drug = nn.Parameter(torch.tensor(0.5))

        self.dropout = nn.Dropout(dropout)

        self.rna_norm = nn.LayerNorm(embedding_dim)
        self.drug_norm = nn.LayerNorm(embedding_dim)

    def forward(self, rna_features, drug_features):

        P_feat = self.rna_feature_encoder(rna_features)
        Q_feat = self.drug_feature_encoder(drug_features)

        rna_gate_weight = torch.softmax(self.rna_gate(rna_features), dim=-1)
        drug_gate_weight = torch.softmax(self.drug_gate(drug_features), dim=-1)
        P_prior = rna_gate_weight @ self.rna_prototypes
        Q_prior = drug_gate_weight @ self.drug_prototypes

        alpha_rna = torch.sigmoid(self.alpha_rna)
        alpha_drug = torch.sigmoid(self.alpha_drug)

        P = (1 - alpha_rna) * P_prior + alpha_rna * P_feat
        Q = (1 - alpha_drug) * Q_prior + alpha_drug * Q_feat

        P = self.dropout(P)
        Q = self.dropout(Q)

        P_norm = self.rna_norm(P)
        Q_norm = self.drug_norm(Q)

        A_recon = torch.matmul(P, Q.t())

        return P_norm, Q_norm, A_recon


class GMF(nn.Module):
    def __init__(self, opt, n_lnc, n_mi, n_drug,
                 rna_in_dim=1280, rna_out_dim=1280, drug_in_dim=768, drug_out_dim=768):
        super(GMF, self).__init__()

        self.opt = opt
        self.n_rna = n_lnc + n_mi
        self.n_drug = n_drug

        if opt.random_emb == 1:
            self.lnc_emb = nn.Parameter(torch.empty(n_lnc, rna_in_dim))
            self.mi_emb = nn.Parameter(torch.empty(n_mi, rna_in_dim))
            self.drug_emb = nn.Parameter(torch.empty(n_drug, drug_in_dim))

            nn.init.xavier_uniform_(self.lnc_emb)
            nn.init.xavier_uniform_(self.mi_emb)
            nn.init.xavier_uniform_(self.drug_emb)

        self.linear = Linear(rna_in_dim, drug_in_dim, opt.gcn_in_dim)

        self.r_gcn_homo = nn.ModuleList([GCNLayer(opt.gcn_in_dim, opt.gcn_out_dim) for _ in range(opt.gcn_layers)])
        self.d_gcn_homo = nn.ModuleList([GCNLayer(opt.gcn_in_dim, opt.gcn_out_dim) for _ in range(opt.gcn_layers)])

        self.gat_hetero = nn.ModuleList([
            PyHeteroGATLayers(
            in_dim=opt.gcn_in_dim,
            out_dim=opt.gcn_out_dim,
            num_heads= opt.num_heads,
            dropout=opt.gat_dropout)
            for _ in range(opt.gat_layers)
        ])

        self.rna_adapter1 = RNAEmbeddingAdapter(in_dim=rna_in_dim, out_dim=rna_out_dim)
        self.rna_adapter2 = RNAEmbeddingAdapter(in_dim=rna_in_dim, out_dim=rna_out_dim)
        self.drug_adapter = DrugEmbeddingAdapter(in_dim=drug_in_dim, out_dim=drug_out_dim)

        self.alpha_score_rna = nn.Parameter(torch.tensor(1.4))
        self.alpha_score_drug = nn.Parameter(torch.tensor(1.4))
        self.predictor1 = Predictor(opt.gcn_out_dim, opt.pred_hid_size)

        self.gmf = MF(
            n_rna=self.n_rna,
            n_drug=n_drug,
            rna_feat_dim=opt.gcn_out_dim,
            drug_feat_dim=opt.gcn_out_dim,
            embedding_dim=opt.gmf_dim,
            dropout=opt.dropout,
            num_prototypes=opt.gmf_num_prototypes
        )

        self.predictor2 = Predictor(opt.gmf_dim, opt.pred_hid_size)

    def forward(self, lnc_emb, mi_emb, drug_emb, rna_sim, drug_sim, train_adj):

        if self.opt.random_emb == 1:
            lnc_emb = self.lnc_emb
            mi_emb = self.mi_emb
            drug_emb = self.drug_emb

        if self.opt.finetune_module == 1:
            lnc_emb = self.rna_adapter1(lnc_emb)
            mi_emb = self.rna_adapter2(mi_emb)
            drug_emb = self.drug_adapter(drug_emb)

        rna_feat, drug_feat = self.linear(lnc_emb, mi_emb, drug_emb)

        if self.opt.gcn_module == 1:

            rna_feat_homo = rna_feat
            for gcn in self.r_gcn_homo.children():
                rna_feat_homo = gcn(rna_feat_homo, rna_sim)

            drug_feat_homo = drug_feat
            for gcn in self.d_gcn_homo.children():
                drug_feat_homo = gcn(drug_feat_homo, drug_sim)

        else:
            rna_feat_homo = torch.zeros_like(rna_feat)
            drug_feat_homo = torch.zeros_like(drug_feat)

        if self.opt.gat_module == 1:

            rna_feat_hetero = rna_feat
            drug_feat_hetero = drug_feat

            for gat_layer in self.gat_hetero:
                rna_feat_hetero, drug_feat_hetero = gat_layer(rna_feat_hetero, drug_feat_hetero, train_adj)

        else:
            rna_feat_hetero = torch.zeros_like(rna_feat)
            drug_feat_hetero = torch.zeros_like(drug_feat)

        alpha1 = torch.sigmoid(self.alpha_score_rna)
        alpha2 = torch.sigmoid(self.alpha_score_drug)

        rna_fusion = alpha1*rna_feat_homo+(1-alpha1) * rna_feat_hetero
        drug_fusion = alpha2 * drug_feat_homo + (1 - alpha2) * drug_feat_hetero

        gnn_rna_feat, gnn_drug_feat = self.predictor1(rna_fusion, drug_fusion)

        if self.opt.gmf_module == 1:

            rna_feat_gmf = rna_feat
            drug_feat_gmf = drug_feat

            gmf_P, gmf_Q, gmf_A_recon = self.gmf(rna_feat_gmf,drug_feat_gmf)

            gmf_rna_feat,gmf_drug_feat= self.predictor2(gmf_P,gmf_Q)

            return (gnn_rna_feat, gnn_drug_feat,
                    gmf_rna_feat, gmf_drug_feat,
                    gmf_A_recon, gmf_P, gmf_Q)

        else:
            return gnn_rna_feat, gnn_drug_feat



