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
import GPUtil

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
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


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

from my_utils.modules import ProteinAngleFlowModule
from my_utils.dataset import MyDataset, ProteinAngleDataModule
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
    if cfg.experiment.debug:
        print("Debug mode.")
        logger = None
    else:
        logger = WandbLogger(project='foldingdiff', log_model=True)
    if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
        print("Logger!")
        logger.experiment.config.update(cfg_dict)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_folder_name = f"/gpfs/radev/scratch/dijk/sh2748/foldingdiff_outputs/output_{current_time}"
    os.makedirs(exp_folder_name, exist_ok=True)
    print(f"Save to {exp_folder_name}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_folder_name,
        filename='{epoch:02d}-{val_loss:.2f}', 
        **cfg.checkpointer
    )

    data_module = ProteinAngleDataModule(data_dir=cfg.data.data_path, batch_size=cfg.data.batch_size)
    protein_angle_flow_module = ProteinAngleFlowModule(cfg=cfg, exp_dir=exp_folder_name)
    
    devices = GPUtil.getAvailable(order='memory', limit = 8)[:cfg.experiment.num_devices]
    trainer = pl.Trainer(logger=logger, callbacks=checkpoint_callback, strategy=DDPStrategy(find_unused_parameters=False), **cfg.trainer)
    trainer.fit(protein_angle_flow_module, data_module, ckpt_path=cfg.experiment.resume_path)

    
            
if __name__ == "__main__":
    main()