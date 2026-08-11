import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from models.model import GMF
import torch.nn.functional as F

class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # warmup: learning rate 从 0 → base_lr 线性增长
        scale = (self.last_epoch + 1) / self.warmup_epochs
        return [base_lr * scale for base_lr in self.base_lrs]


class MaskedBCELoss(nn.BCELoss):
    def forward(self, new_p_feat, new_d_feat, adj, train_mask, test_mask):

        self.reduction = "none"

        cosine_sim = F.cosine_similarity(new_p_feat.unsqueeze(1), new_d_feat.unsqueeze(0), dim=2)
        cosine_sim_exp = torch.exp(cosine_sim / 0.5)

        sim_num = adj * cosine_sim_exp * train_mask

        sim_diff = cosine_sim_exp * (1 - adj) * train_mask  #Nr*Nd
        sim_diff_sum = torch.sum(sim_diff, dim=1)  #Nr
        sim_diff_sum_expend = sim_diff_sum.repeat(new_d_feat.shape[0], 1).T  #Nr*Nd

        sim_den = sim_num + sim_diff_sum_expend

        loss = torch.div(sim_num, sim_den)

        loss1 = torch.clamp(1 - adj + 1 - train_mask, max=1) + loss
        loss_log = -torch.log(loss1)

        pred = F.sigmoid(new_p_feat.mm(new_d_feat.t()))
        unmasked_loss = super(MaskedBCELoss, self).forward(pred, adj)

        loss_c = loss_log.mean()
        loss_b = (unmasked_loss * train_mask).sum() / train_mask.sum()

        train_loss = loss_b + loss_c
        test_loss = (unmasked_loss * test_mask).sum() / test_mask.sum()

        return train_loss, test_loss, pred


class GMFReconstructionLoss(nn.Module):

    def __init__(self, lambda_reg=0.01):
        super(GMFReconstructionLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, A_recon, A_true, train_mask, P, Q):

        A_recon_sigmoid = torch.sigmoid(A_recon)
        bce = F.binary_cross_entropy(A_recon_sigmoid, A_true, reduction='none')
        recon_loss = (bce * train_mask).sum() / train_mask.sum()

        reg_loss = self.lambda_reg * (
                torch.norm(P, p='fro') ** 2 + torch.norm(Q, p='fro') ** 2
        ) / (P.shape[0] + Q.shape[0])

        total_loss = recon_loss + reg_loss

        return total_loss, recon_loss


class Trainer(BaseModel):
    def name(self):
        return 'GMF Trainer'

    def __init__(self, opt,n_lnc, n_mi, n_drug,
                 rna_in_dim, rna_out_dim, drug_in_dim, drug_out_dim):
        super(Trainer, self).__init__(opt)
        self.opt = opt
        self.device = torch.device(opt.device)

        self.model = GMF(opt, n_lnc, n_mi, n_drug,
                         rna_in_dim, rna_out_dim, drug_in_dim, drug_out_dim
                         ).to(self.device)

        init_weights(self.model, init_type=opt.init_type, gain=opt.init_gain)

        self.optimizer = create_optimizer(self.opt, model=self.model)
        self.lr_scheduler, _ = create_scheduler(self.opt, self.optimizer)

        self.criterion_bce = MaskedBCELoss()
        self.criterion_gmf = GMFReconstructionLoss(lambda_reg=opt.gmf_reg)

    def set_input(self, data):
        self.lnc_emb = data.lnc_dmap.to(self.device)
        self.mi_emb = data.mi_dmap.to(self.device)
        self.drug_emb = data.drug_dmap.to(self.device)

        self.rna_sim = data.rna_sim.to(self.device)
        self.drug_sim = data.drug_sim.to(self.device)

        self.adj = data.adj.to(self.device)

        self.train_adj = data.train_adj.to(self.device)

        self.train_mask = data.train_mask.to(self.device)
        self.test_mask = data.test_mask.to(self.device)

    def forward(self):
        if self.opt.gmf_module == 1:
            (self.gnn_rna_feat, self.gnn_drug_feat,
             self.gmf_rna_feat, self.gmf_drug_feat,
             self.gmf_A_recon, self.gmf_P, self.gmf_Q) = \
                self.model(self.lnc_emb, self.mi_emb, self.drug_emb,
                           self.rna_sim, self.drug_sim, self.train_adj)

        else:
            self.gnn_rna_feat, self.gnn_drug_feat = (
                self.model(self.lnc_emb, self.mi_emb, self.drug_emb,
                           self.rna_sim, self.drug_sim, self.train_adj))

    def optimize_parameters(self):
        self.forward()
        if self.opt.gmf_module == 1:

            final_rna_feat = torch.cat([self.gnn_rna_feat, self.gmf_rna_feat], dim=-1)
            final_drug_feat = torch.cat([self.gnn_drug_feat, self.gmf_drug_feat], dim=-1)

            train_loss_final, test_loss_final, pred_final = self.criterion_bce(
                final_rna_feat, final_drug_feat, self.adj,
                self.train_mask, self.test_mask
            )

            gmf_recon_loss, gmf_recon_only = self.criterion_gmf(
                self.gmf_A_recon, self.adj, self.train_mask,
                self.gmf_P, self.gmf_Q
            )

            w_final = 1.0
            w_recon = self.opt.gmf_recon_weight

            self.train_loss = (
                    w_final * train_loss_final +
                    w_recon * gmf_recon_loss
            )

            self.test_loss = (
                    w_final * test_loss_final
            )
            self.loss_components = {
                'final': train_loss_final.item(),
                'gmf_recon': gmf_recon_only.item(),
            }

            self.pred = pred_final

        else:
            self.train_loss, self.test_loss, self.pred = self.criterion_bce(
                self.gnn_rna_feat, self.gnn_drug_feat,
                self.adj, self.train_mask, self.test_mask
            )


        self.optimizer.zero_grad()
        self.train_loss.backward()

        self.optimizer.step()


    def lr_step(self, epoch):
        self.lr_scheduler.step(epoch)
