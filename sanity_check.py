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

from foldingdiff import datasets
from foldingdiff import modelling
from foldingdiff import losses
from foldingdiff import beta_schedules
from foldingdiff import plotting
from foldingdiff import utils
from foldingdiff import custom_metrics as cm
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

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.1):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

import torch
from torch import nn
from torch.distributions import Normal
from ipdb import set_trace

# Copied from DanielFLevine/ifm/utils/modules.py

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.1):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        identity = x
        out = self.linear(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        # out += identity  # Skip connection
        return out

class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.1):
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        identity = x
        out = self.linear(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class MidFC(nn.Module):
    def __init__(self, dim, num_layers, dropout_prob=0.1):
        super(MidFC, self).__init__()
        self.layers = nn.Sequential(
            *[MLPLayer(dim, dim) for _ in range(num_layers)]
        )
        
    def forward(self, x):
        return self.layers(x)


class CustomDecoder(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, num_blocks=1):
        super(CustomDecoder, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(input_dim=hidden_dim, output_dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.residual_blocks(x)
        x = self.final_layer(x)
        return x

class CustomVAEDecoder(nn.Module):
    def __init__(self, input_dim, vae_latent_dim, decoder_hidden_dim, output_dim, num_blocks=1):
        super(CustomVAEDecoder, self).__init__()
        self.mean_encoder = ResidualBlock(input_dim=input_dim, output_dim=vae_latent_dim)
        self.var_encoder = ResidualBlock(input_dim=input_dim, output_dim=vae_latent_dim)
        self.var_activation = torch.exp
        self.var_eps = 0.0001
        self.decoder = CustomDecoder(
            input_dim=vae_latent_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=output_dim,
            num_blocks=num_blocks
        )

    def forward(self, x, temperature=1.0):
        '''
            Returns:
                outputs:    output representations
                latents:    VAE latent representations
                dist:       a torch.dist object used to compute KL divergence
        '''
        # set_trace()
        mu = self.mean_encoder(x)
        var = self.var_encoder(x)
        var = self.var_activation(var) + self.var_eps
        dist = Normal(mu, temperature*(var.sqrt()))
        latents = dist.rsample()
        outputs = self.decoder(latents)
        # set_trace()
        return outputs, latents, dist

from transformers import GPT2Model, GPT2Config
class ProteinAngleFlowModel(nn.Module):
    def __init__(self, 
        input_size, 
        proj_hid_size,
        llm_embd_size,
        num_res_per_group=1,
        llm_name="gpt2",
        use_pretrained_weights=True,
        use_custom_gpt2_arch=False,
        llm_n_layer=1,
        llm_n_head=1,
        vae_latent_dim=128,
        decoder_hidden_dim=256,
    ):
        super().__init__()
        self.num_res_per_group = num_res_per_group
        self.proj_in = SimpleMLP(input_size=input_size, hidden_size=proj_hid_size, output_size=int(llm_embd_size/num_res_per_group))
        self._create_llm(llm_name, use_custom_gpt2_arch, use_pretrained_weights, llm_embd_size, llm_n_layer, llm_n_head)
        self.vae_decoder = CustomVAEDecoder(input_dim=llm_embd_size, vae_latent_dim=vae_latent_dim, decoder_hidden_dim=decoder_hidden_dim, output_dim=input_size*num_res_per_group)
    
    def _create_llm(self, llm_name, use_custom_gpt2_arch, use_pretrained_weights, llm_embd_size, llm_n_layer, llm_n_head):
        if llm_name == "gpt2": 
            print("Use GPT2 as LLM.")
            if not use_custom_gpt2_arch:
                print("Use GPT2 model with pretrained architecture!")
                if llm_embd_size != 768:
                    print(f"Override GPT2 dimension from {self._gpt_conf.n_embd} to 768!")
                if use_pretrained_weights:
                    self.llm_model = GPT2Model.from_pretrained('gpt2') 
                    print("Use GPT2 model with PRETRAINED weights!")
                else:
                    config = GPT2Config()  # Default GPT-2 configuration
                    self.llm_model = GPT2Model(config=config)
                    print("Use GPT2 model with RANDOM weights!")
            else:
                config = GPT2Config(
                    n_embd=llm_embd_size,
                    n_layer=llm_n_layer,
                    n_head=llm_n_head
                )
                self.llm_model = GPT2Model(config=config)
                print("Use CUSTOM GPT2 architecture!")
                print("Use GPT2 model with RANDOM weights!")
            self.llm_model.wte = None
        elif self.llm_name in ["pythia-160m", 'pythia-410m']:
            llm_name = 'EleutherAI/' + self._model_conf.llm_name
            print(f"Use {llm_name} as LLM.")
            if use_pretrained_weights:
                print(f"Use {llm_name} model with PRETRAINED weights!")
                self.llm_model = AutoModel.from_pretrained(llm_name)
            else:
                pythia_config = AutoConfig.from_pretrained(llm_name)
                self.llm_model = AutoModel.from_config(pythia_config)
                print(f"Use {llm_name} model with RANDOM weights!")

    def forward(self, protein_angles):
        residue_embeds = self.proj_in(protein_angles) # [b, t, s, d]
        _b, _t, _s, _d = residue_embeds.shape
        group_embeds = rearrange(residue_embeds, 'b t (s0 g) d -> b (t s0) (g d)', g = self.num_res_per_group) # [b, t*s/g, g*d], g: num_res_per_group
        llm_outputs = self.llm_model(inputs_embeds=group_embeds)
        llm_outputs_embeds = llm_outputs.last_hidden_state
        vae_outputs, vae_latents, vae_dist = self.vae_decoder(llm_outputs_embeds)
        outputs_embeds = rearrange(vae_outputs, 'b (t s0) (g d) -> b t (s0 g) d', s0=int(_s/self.num_res_per_group),g = self.num_res_per_group) # [b, t, s, d], g: num_res_per_group
        outputs_embeds = modulo_with_wrapped_range(outputs_embeds)
        # KL divergence loss
        z_cond_dist = vae_dist
        z_prior_dist = Normal(torch.zeros_like(vae_latents), torch.ones_like(vae_latents))
        # set_trace()
        kl_divergence_z = kl_divergence(
                        z_cond_dist,
                        z_prior_dist
                    )
        kl_divergence_z = rearrange(kl_divergence_z, 'b (t s0) (g d) -> b t (s0 g) d', s0=int(_s/self.num_res_per_group), g=self.num_res_per_group) #[b, t, s, d]
        kl_divergence_z = kl_divergence_z.sum(dim=-1)[:,:-1,:] # throw away the last time point
        return {
            'output_embeds': outputs_embeds,
            'kl_divergence': kl_divergence_z.mean()
        }
    
    def generate_next_tokens(self, protein_angles, temperature=1):
        assert not temperature < 0, "Temperature should be non-negative!"
        residue_embeds = self.proj_in(protein_angles) # [b, t, s, d]
        _b, _t, _s, _d = residue_embeds.shape
        group_embeds = rearrange(residue_embeds, 'b t (s0 g) d -> b (t s0) (g d)', g = self.num_res_per_group) # [b, t*s/g, g*d], g: num_res_per_group
        llm_outputs = self.llm_model(inputs_embeds=group_embeds)
        llm_outputs_embeds = llm_outputs.last_hidden_state
        vae_outputs, vae_latents, vae_dist = self.vae_decoder(llm_outputs_embeds, temperature=temperature)
        outputs_embeds = rearrange(vae_outputs, 'b (t s0) (g d) -> b t (s0 g) d', s0=int(_s/self.num_res_per_group),g = self.num_res_per_group) # [b, t, s, d], g: num_res_per_group
        outputs_embeds = modulo_with_wrapped_range(outputs_embeds)
        
        return outputs_embeds[:,-1,:,:].unsqueeze(1)

    def generate(self, input_ids, temperature=1, max_length=8):
        output_sequences = input_ids
        while output_sequences.shape[1] < max_length:
            next_tokens = self.generate_next_tokens(output_sequences)
            output_sequences = torch.cat([output_sequences, next_tokens], dim=1)
        return output_sequences



def modulo_with_wrapped_range(
    vals, range_min: float = -np.pi, range_max: float = np.pi
):
    """
    Modulo with wrapped range -- capable of handing a range with a negative min

    modulo_with_wrapped_range(3, -2, 2)
    -1
    """
    assert range_min <= 0.0
    assert range_min < range_max

    # Modulo after we shift values
    top_end = range_max - range_min
    # Shift the values to be in the range [0, top_end)
    vals_shifted = vals - range_min
    # Perform modulo
    vals_shifted_mod = vals_shifted % top_end
    # Shift back down
    retval = vals_shifted_mod + range_min

    # Checks
    # print("Mod return", vals, " --> ", retval)
    # if isinstance(retval, torch.Tensor):
    #     notnan_idx = ~torch.isnan(retval)
    #     assert torch.all(retval[notnan_idx] >= range_min)
    #     assert torch.all(retval[notnan_idx] < range_max)
    # else:
    #     assert (
    #         np.nanmin(retval) >= range_min
    #     ), f"Illegal value: {np.nanmin(retval)} < {range_min}"
    #     assert (
    #         np.nanmax(retval) <= range_max
    #     ), f"Illegal value: {np.nanmax(retval)} > {range_max}"
    return retval




# Define a PyTorch Dataset class
class MyDataset(Dataset):
    def __init__(self, dataframe=None, data_path=None):
        if data_path is not None:
            print(f"Initializing from a data path: {data_path}")
            self.data = torch.load(data_path)
        elif data_path is None and dataframe is not None:
            print("Initializing from a dataframe.")
            self.data = torch.stack(dataframe['angles'].tolist())
        else:
            raise Exception("Dataframe and data path are both None!")
        
         
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


lr = 0.000005
weight_decay = 0
beta = 0.01
num_epochs=1000
ckpt_freq=100
batch_size=32


# Initialize the dataset with the DataFrame
dataset = MyDataset(data_path="protein_angles_100sanity_128.pt")
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,       # Number of samples per batch
    shuffle=True,        # Shuffle the data at every epoch
    num_workers=1        # Number of subprocesses to use for data loading
)

from IntegralFlowMatching.integral_flow_matching import IntegralFlowMatcher
IFM = IntegralFlowMatcher(sigma=0.1, same_time=True, time_interp=True, noise_on_x0=True, noise_on_x1=True)

protein_angle_model = ProteinAngleFlowModel(input_size=6, proj_hid_size=128, llm_embd_size=768).to("cuda")


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_folder_name = f"outputs/output_{current_time}"
os.makedirs(exp_folder_name, exist_ok=True)

wandb.init(project="foldingdiff", entity="sizhuang")
wandb.log({
    'lr': lr,
    'weight_decay': weight_decay,
    'beta': beta,
    'num_epochs': num_epochs
})

optimizer = torch.optim.Adam(protein_angle_model.parameters(), lr=lr, weight_decay=weight_decay)

from tqdm import tqdm

protein_angle_model = protein_angle_model.to("cuda")
losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in tqdm(dataloader):
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
        loss = mse_loss + beta * kl_div
        wandb.log({"train_loss": loss.item(),
            "kl_divergence": beta * kl_div,
            "mse_loss": mse_loss
            })
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        epoch_loss += loss.item()
    wandb.log({'epoch': epoch})
    
    if (epoch+1) % ckpt_freq == 0:
        total_preds = []
        epoch_folder_name=f"epoch_{epoch+1}"
        for i in tqdm(range(5)): # 5 is HARD CODED for now!!!
            x0 = torch.randn(20, 128, 6).unsqueeze(1).to(torch.float).to("cuda") # HARD CODED for now!!!
            preds = protein_angle_model.generate(x0)[:, -1, :, :].cpu().detach()
            total_preds.append(preds)
        set_trace()
        total_preds = torch.cat(total_preds, dim=0).reshape(-1, 6)
        angles = torch.load("protein_angles_100sanity_128.pt")
        
        folder_name = os.path.join(exp_folder_name, epoch_folder_name)
        os.makedirs(folder_name, exist_ok=True)
        for i in range(6):
            _ = plt.hist(angles.reshape(-1, 6)[:,i], bins=100, range=(-np.pi, np.pi), label="GT angles")
            _ = plt.hist(total_preds.reshape(-1, 6)[:,i], bins=100, range=(-np.pi, np.pi), label="Predicted angles")
            
            plt.legend()
            plt.savefig(os.path.join(folder_name, f"angle_{i}_histogram.png"))
            plt.close()
        torch.save(protein_angle_model.state_dict(), os.path.join(folder_name, 'model_state_dict.pth'))



    print("Epoch {}: loss: {}".format(epoch+1, epoch_loss))
torch.save(protein_angle_model.state_dict(), os.path.join(exp_folder_name, "final_model_state_dict.pth"))


wandb.finish()
        
