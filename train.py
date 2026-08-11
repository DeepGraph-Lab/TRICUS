from option.train_options import TrainOptions
from networks.trainer import Trainer
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
import torch
from utils.utils import seed_everything, Logger
from dataclasses import dataclass


@dataclass
class DatasetResult:
    adj: torch.Tensor
    train_adj: torch.Tensor
    rna_sim: torch.Tensor
    drug_sim: torch.Tensor
    lnc_dmap: torch.Tensor
    mi_dmap: torch.Tensor
    drug_dmap: torch.Tensor
    train_mask: torch.Tensor
    test_mask: torch.Tensor
    pos_test_ij_tensor: torch.Tensor
    unlabelled_test_ij_tensor: torch.Tensor
    n_lnc: int
    n_mi: int
    n_drug: int



def read_data(i) -> DatasetResult:
    adj_np = pd.read_csv(r"data/ncrna-drug_split.csv", index_col=0).values

    lnc_dmap_np = np.load("data/lncRNA_embeddings_RiMALMo.npy")
    mi_dmap_np = np.load("data/miRNA_embeddings_RiMALMo.npy")
    drug_dmap_np = np.load("data/drug_embeddings.npy")

    with open(r"fold_info.pickle", "rb") as f:
        fold_info = pickle.load(f)

    pos_train_ij_list = fold_info["pos_train_ij_list"]
    pos_test_ij_list = fold_info["pos_test_ij_list"]
    unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
    unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]

    p_gip_list = fold_info["p_gip_list"]
    d_gip_list = fold_info["d_gip_list"]

    lnc_dmap = torch.FloatTensor(lnc_dmap_np)
    mi_dmap = torch.FloatTensor(mi_dmap_np)
    drug_dmap = torch.FloatTensor(drug_dmap_np)
    adj = torch.FloatTensor(adj_np)

    pos_train_ij = pos_train_ij_list[i]
    pos_test_ij = pos_test_ij_list[i]
    unlabelled_train_ij = unlabelled_train_ij_list[i]
    unlabelled_test_ij = unlabelled_test_ij_list[i]
    np.random.shuffle(unlabelled_test_ij)

    train_mask_np = np.zeros_like(adj_np)
    train_mask_np[tuple(list(pos_train_ij.T))] = 1
    train_mask_np[tuple(list(unlabelled_train_ij.T))] = 1

    test_mask_np = np.zeros_like(adj_np)
    test_mask_np[tuple(list(pos_test_ij.T))] = 1
    test_mask_np[tuple(list(unlabelled_test_ij.T))] = 1


    train_adj_np = np.zeros_like(adj_np)
    train_adj_np[tuple(pos_train_ij.T)] = 1
    train_adj = torch.FloatTensor(train_adj_np)

    rna_sim_np = p_gip_list[i]
    drug_sim_np = d_gip_list[i]

    np.fill_diagonal(rna_sim_np, 1)
    np.fill_diagonal(drug_sim_np, 1)

    rna_sim = torch.FloatTensor(rna_sim_np)
    drug_sim = torch.FloatTensor(drug_sim_np)

    train_mask = torch.FloatTensor(train_mask_np)
    test_mask = torch.FloatTensor(test_mask_np)
    pos_test_ij_tensor = torch.IntTensor(pos_test_ij)
    unlabelled_test_ij_tensor = torch.IntTensor(unlabelled_test_ij)

    n_lnc = lnc_dmap_np.shape[0]
    n_mi = mi_dmap_np.shape[0]
    n_drug = drug_dmap_np.shape[0]

    return DatasetResult(
        adj=adj,
        train_adj=train_adj,
        rna_sim=rna_sim,
        drug_sim=drug_sim,
        lnc_dmap=lnc_dmap,
        mi_dmap=mi_dmap,
        drug_dmap=drug_dmap,
        train_mask=train_mask,
        test_mask=test_mask,
        pos_test_ij_tensor=pos_test_ij_tensor,
        unlabelled_test_ij_tensor=unlabelled_test_ij_tensor,
        n_lnc=n_lnc,
        n_mi=n_mi,
        n_drug=n_drug
    )


if __name__ == '__main__':
    seed_everything(42)
    opt = TrainOptions().parse()

    data_sample = read_data(0)

    model = Trainer(
        opt,
        n_lnc=data_sample.n_lnc,
        n_mi=data_sample.n_mi,
        n_drug=data_sample.n_drug,
        rna_in_dim=1280,
        rna_out_dim=1280,
        drug_in_dim=768,
        drug_out_dim=768
    )
    model.cuda()
    logger = Logger(5)

    for i in tqdm(range(5), desc="Fold"):

        print(f"\n========== Fold {i} ==========")
        data = read_data(i)
        test_idx = torch.argwhere(data.test_mask == 1)

        for epoch in tqdm(range(opt.niter), desc="Epoch"):
            model.set_input(data)
            model.optimize_parameters()

            logger.update(
                i, epoch, model.adj, model.pred, test_idx,
                model.train_loss.item(),
                model.test_loss.item(),
                data.pos_test_ij_tensor,
                data.unlabelled_test_ij_tensor
            )

            model.lr_step(epoch)


    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger.save(opt.checkpoints_dir, opt.name)
    print(f"\n✅ Training complete. Results saved to: {save_dir}")




