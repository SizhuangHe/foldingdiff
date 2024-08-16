import os, sys
import shutil
import json
import logging
from pathlib import Path
import multiprocessing
import argparse
import functools
from datetime import datetime
from typing import *
from ipdb import set_trace

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Normal, kl_divergence

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from transformers import BertConfig
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from foldingdiff import datasets
from foldingdiff import modelling
from foldingdiff import losses
from foldingdiff import beta_schedules
from foldingdiff import plotting
from foldingdiff import utils
from foldingdiff import custom_metrics as cm

from my_utils.modules import ProteinAngleFlowModel
from my_utils.dataset import MyDataset
from my_utils.utils import modulo_with_wrapped_range
from IntegralFlowMatching.integral_flow_matching import IntegralFlowMatcher
import argparse
import wandb
wandb.login()

assert torch.cuda.is_available(), "Requires CUDA to train"
# reproducibility
torch.manual_seed(6489)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

# Define some typing literals
ANGLES_DEFINITIONS = Literal[
    "canonical", "canonical-full-angles", "canonical-minimal-angles", "cart-coords"
]

@hydra.main(config_path="my_configs", config_name="train")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project="foldingdiff", entity="sizhuang", config=cfg_dict)
    
    data_cfg = cfg.data
    ifm_cfg = cfg.ifm
    model_cfg = cfg.model
    vae_cfg = cfg.model.vae
    exp_cfg = cfg.experiment

    dataset = MyDataset(data_path=data_cfg.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,       # Number of samples per batch
        shuffle=True,        # Shuffle the data at every epoch
        num_workers=1        # Number of subprocesses to use for data loading
    )
    
    IFM = IntegralFlowMatcher(sigma=ifm_cfg.sigma, same_time=True, time_interp=True, noise_on_x0=True, noise_on_x1=True)

    protein_angle_model = ProteinAngleFlowModel(input_size=6, proj_hid_size=model_cfg.proj_hid_size, llm_embd_size=768, vae_latent_dim=vae_cfg.vae_latent_dim, decoder_hidden_dim=vae_cfg.decoder_hidden_dim).to("cuda")
    protein_angle_model = protein_angle_model.to("cuda")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_folder_name = f"outputs/output_{current_time}"
    os.makedirs(exp_folder_name, exist_ok=True)

    optimizer = torch.optim.Adam(protein_angle_model.parameters(), lr=exp_cfg.lr, weight_decay=exp_cfg.weight_decay)

    losses = []
    for epoch in range(exp_cfg.num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            batch = batch.to("cuda")
            x0 = torch.randn_like(batch)
            t, xt, mu_t = IFM.sample_conditional_flow(rearrange(x0, 'b s d -> b (s d)').to("cuda"), rearrange(batch, 'b s d -> b (s d)').to("cuda"), 8,  device = "cuda")
            xt = rearrange(xt, 'b t (s d) -> b t s d', s=x0.shape[1])
            xt = modulo_with_wrapped_range(xt)

            xt = xt.to(torch.float32)
        
            outputs = protein_angle_model(xt)
            shifted_pred_tokens = outputs["output_embeds"][:, :-1, ...]
            shifted_gt_tokens = xt[:,1:,...]

            mse_loss = F.mse_loss(shifted_pred_tokens, shifted_gt_tokens, reduction="none").sum(dim=-1).mean()
            kl_div = outputs['kl_divergence']
            loss = mse_loss + exp_cfg.beta * kl_div
            wandb.log({"train_loss": loss.item(),
                "kl_divergence": exp_cfg.beta * kl_div,
                "mse_loss": mse_loss
                })
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            epoch_loss += loss.item()
        wandb.log({'epoch': epoch})
        
        if (epoch+1) % exp_cfg.ckpt_freq == 0:
            total_preds = []
            epoch_folder_name=f"epoch_{epoch+1}"
            for i in range(10): # 5 is HARD CODED for now!!!
                x0 = torch.randn(10, 128, 6).unsqueeze(1).to(torch.float).to("cuda") # HARD CODED for now!!!
                preds = protein_angle_model.generate(x0)[:, -1, :, :].cpu().detach()
                total_preds.append(preds)
            # set_trace()
            total_preds = torch.cat(total_preds, dim=0).reshape(-1, 6)
            angles = torch.load("protein_angles_100sanity_128.pt")
            
            folder_name = os.path.join(exp_folder_name, epoch_folder_name)
            os.makedirs(folder_name, exist_ok=True)
            for i in range(6):
                _ = plt.hist(angles.reshape(-1, 6)[:,i], bins=100, range=(-np.pi, np.pi), label="GT angles")
                _ = plt.hist(total_preds.reshape(-1, 6)[:,i], bins=100, range=(-np.pi, np.pi), label="Predicted angles")
                
                plt.legend()
                plt.savefig(os.path.join(folder_name, f"angle_{i}_histogram.png"))
                wandb.log({f"angle_{i}_histogram": wandb.Image(os.path.join(folder_name, f"angle_{i}_histogram.png"))})
                plt.close()
            torch.save(protein_angle_model.state_dict(), os.path.join(folder_name, 'model_state_dict.pth'))



        print("Epoch {}: loss: {}".format(epoch+1, epoch_loss))
    torch.save(protein_angle_model.state_dict(), os.path.join(exp_folder_name, "final_model_state_dict.pth"))


    wandb.finish()
            
if __name__ == "__main__":
    main()