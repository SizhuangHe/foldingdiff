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
from my_utils.dataset import ProteinAngleDataModule
from my_utils.utils import modulo_with_wrapped_range
from IntegralFlowMatching.integral_flow_matching import IntegralFlowMatcher
import argparse
import wandb
wandb.login()

assert torch.cuda.is_available(), "Requires CUDA to infer"
# reproducibility
torch.manual_seed(6489)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
@hydra.main(config_path="my_configs", config_name="inference")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_folder_name = f"/gpfs/radev/scratch/dijk/sh2748/foldingdiff_inference_outputs/output_{current_time}"
    os.makedirs(exp_folder_name, exist_ok=True)
    print(f"Save to {exp_folder_name}")
    if cfg.experiment.debug:
        print("Debug mode.")
        logger = None
    else:
        logger = WandbLogger(**cfg.wandb)
    if logger is not None and isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
        print("Logger!")
        logger.experiment.config.update(cfg_dict)

    data_module = ProteinAngleDataModule()
    predict_dataloader = data_module.predict_dataloader(num_samples_per_length=10, min_length=60, max_length=128)
    protein_angle_flow_module = ProteinAngleFlowModule.load_from_checkpoint(
        checkpoint_path=cfg.inference.ckpt_path,
        map_location=lambda storage, loc: storage.cuda(0)
    )
    protein_angle_flow_module.exp_dir = exp_folder_name
    protein_angle_flow_module.inference_temperature = cfg.inference.temperature
    # set_trace()
    devices = GPUtil.getAvailable(order='memory', limit = 8)[:cfg.experiment.num_devices]
    trainer = pl.Trainer(devices=devices, logger=logger, **cfg.trainer)
    trainer.predict(protein_angle_flow_module, predict_dataloader)

    
            
if __name__ == "__main__":
    main()