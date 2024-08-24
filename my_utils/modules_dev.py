import os
import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from einops import rearrange
from torch.distributions import Normal, kl_divergence
from .utils import modulo_with_wrapped_range
import pytorch_lightning as pl
import torch.nn.functional as F
from IntegralFlowMatching.integral_flow_matching import IntegralFlowMatcher
from ipdb import set_trace
import pandas as pd
import multiprocessing
from foldingdiff.angles_and_coords import create_new_chain_nerf
from tqdm import tqdm
from pytorch_lightning.loggers import WandbLogger

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
    def __init__(self, model_cfg):
        super().__init__()
        llm_cfg = model_cfg.llm
        vae_cfg = model_cfg.vae
        self.model_cfg=model_cfg
        self.num_res_per_group = llm_cfg.num_res_per_group
        self.proj_in = SimpleMLP(input_size=model_cfg.input_size, hidden_size=model_cfg.proj_hid_size, output_size=int(llm_cfg.llm_embd_size/llm_cfg.num_res_per_group))
        self._create_llm(llm_cfg.llm_name, llm_cfg.use_custom_gpt2_arch, llm_cfg.use_pretrained_weights, llm_cfg.use_flash_attn,llm_cfg.llm_embd_size, llm_cfg.llm_n_layer, llm_cfg.llm_n_head)
        self.vae_decoder = CustomVAEDecoder(input_dim=llm_cfg.llm_embd_size, vae_latent_dim=vae_cfg.vae_latent_dim, decoder_hidden_dim=vae_cfg.decoder_hidden_dim, output_dim=model_cfg.input_size*llm_cfg.num_res_per_group)
        if self.model_cfg.global_pe:
            self.global_pe_encoder = SimpleMLP(input_size=1, hidden_size=model_cfg.proj_hid_size, output_size=llm_cfg.llm_embd_size)

    def _create_llm(self, llm_name, use_custom_gpt2_arch, use_pretrained_weights, use_flash_attn, llm_embd_size, llm_n_layer, llm_n_head):
        if llm_name == "gpt2": 
            print("Use GPT2 as LLM.")
            if not use_custom_gpt2_arch:
                print("Use GPT2 model with pretrained architecture!")
                if llm_embd_size != 768:
                    print(f"Override GPT2 dimension from {self._gpt_conf.n_embd} to 768!")
                if use_pretrained_weights:
                    if use_flash_attn:
                        print("Use flash attention!")
                        self.llm_model = GPT2Model.from_pretrained('gpt2', attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
                    else:
                        print("Not using flash attention!")
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

    def _global_pe(self, batch_size, n_residues, device, dtype):
        base_tensor = torch.arange(1, self.model_cfg.n_timepoints + 1, device=device, dtype=dtype)
        return base_tensor.repeat_interleave(int(n_residues/self.num_res_per_group)).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1) # [b, t*s/g, 1]

    def forward(self, protein_angles, protein_length, device):
        group_embeds = self.proj_in(protein_angles) # [b, t*s/g, g*d], g: num_res_per_group
        _b = group_embeds.shape[0]
        _s = protein_length
        if self.model_cfg.global_pe:
            # set_trace()
            time_encoding = self._global_pe(_b, _s, device, dtype=group_embeds.dtype)
            time_encoding = self.global_pe_encoder(time_encoding)
            group_embeds = group_embeds + time_encoding
   
        llm_outputs = self.llm_model(inputs_embeds=group_embeds)
        llm_outputs_embeds = llm_outputs.last_hidden_state
        vae_outputs, vae_latents, vae_dist = self.vae_decoder(llm_outputs_embeds)
        outputs_embeds = vae_outputs
        outputs_embeds = modulo_with_wrapped_range(outputs_embeds) # [b, t*s/g, 6]
        # KL divergence loss
        z_cond_dist = vae_dist
        z_prior_dist = Normal(torch.zeros_like(vae_latents), torch.ones_like(vae_latents))
        # set_trace()

        kl_divergence_z = kl_divergence(
                        z_cond_dist,
                        z_prior_dist
                    ) # [b, t*s/g, d_vae]
        kl_divergence_z = kl_divergence_z.sum(dim=-1)[:,:-1] # throw away the last time point
        # set_trace()
        return {
            'output_embeds': outputs_embeds,
            'kl_divergence': kl_divergence_z
        }

    def generate(self, input_ids, device, temperature=1, max_length=8):
        print(f"Generation temperature is {temperature}")
        output_sequences = input_ids # input_ids is x0, [b, 1, s, 6]
        # set_trace()
        _b, _, _s, _ = output_sequences.shape
        output_sequences = rearrange(output_sequences, 'b t (s0 g) d -> b (t s0) (g d)', g = self.num_res_per_group) #[b, t*s/g, g*d], currently it's [b, s, 1]
        timesteps = torch.ones(_b, _s, device=device, dtype=output_sequences.dtype).unsqueeze(-1) # [b, s, 1], the initial timesteps, full of 1's, may not work when g > 1
        pbar = tqdm(total=max_length * _s - output_sequences.shape[1])
        while output_sequences.shape[1] < max_length * _s:
            group_embeds = self.proj_in(output_sequences) # [b, t*s/g, g*d]
            
            if self.model_cfg.global_pe:
                time_encoding = self.global_pe_encoder(timesteps)
                group_embeds = group_embeds + time_encoding

            llm_outputs = self.llm_model(inputs_embeds=group_embeds)
            llm_outputs_embeds = llm_outputs.last_hidden_state
            vae_outputs, vae_latents, vae_dist = self.vae_decoder(llm_outputs_embeds, temperature=temperature)
            
            outputs_embeds = modulo_with_wrapped_range(vae_outputs)
            output_sequences = torch.cat([output_sequences, outputs_embeds[:,-1,:][:, None, :]], dim=1)
            next_timestep = torch.full((_b,1,1), timesteps.shape[1] // _s + 1, device=device, dtype=output_sequences.dtype)
            timesteps = torch.cat([timesteps, next_timestep], dim=1)
            pbar.update(1)
        pbar.close()
        return output_sequences # [b, t*s/g, g*d]

class ProteinAngleFlowModule(pl.LightningModule):
    def __init__(self, cfg, exp_dir):
        super().__init__()
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.model_cfg = cfg.model
        self.exp_cfg = cfg.experiment
        ifm_cfg = cfg.ifm
        self.model = ProteinAngleFlowModel(model_cfg=self.model_cfg)
        self.integral_flow_matcher = IntegralFlowMatcher(sigma=ifm_cfg.sigma, same_time=True, time_interp=True, noise_on_x0=True, noise_on_x1=True)
        self.inference_temperature = 1
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            **self.exp_cfg.optimizer
        )
        if self.exp_cfg.lr_scheduler is None:
            return {
                "optimizer": optimizer,
            }
        elif self.exp_cfg.lr_scheduler == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.exp_cfg.optimizer.lr, steps_per_epoch=1, epochs=self.cfg.trainer.max_epochs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        elif self.exp_cfg.lr_scheduler == "cos_ann":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.trainer.max_epochs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        else:
            raise NotImplementedError
        # return torch.optim.Adam(self.model.parameters(), lr=self.cfg.experiment.lr, weight_decay=self.cfg.experiment.weight_decay)

    def model_step(self, batch):
        batch_angles, length = batch
        max_length = batch_angles.shape[1]
        length = length[0].item()
        batch_angles = batch_angles[:, :length, :] # discard existing pads
        
        x0 = modulo_with_wrapped_range(torch.randn_like(batch_angles))
        t, xt, mu_t = self.integral_flow_matcher.sample_conditional_flow(rearrange(x0, 'b s d -> b (s d)').to("cuda"), rearrange(batch_angles, 'b s d -> b (s d)').to("cuda"), self.model_cfg.n_timepoints,  device = "cuda")
        xt = rearrange(xt, 'b t (s d) -> b t s d', s=x0.shape[1])
        xt = modulo_with_wrapped_range(xt).to(x0.dtype) #[b, t, length, 6]

        # set_trace()
        xt = rearrange(xt, 'b t (s0 g) d -> b (t s0) (g d)', g = self.model_cfg.llm.num_res_per_group) # [b, t*s/g, 6*g]
        
        outputs = self.model(xt, length, device=self.device)

        shifted_pred_tokens = outputs["output_embeds"][:, :-1, ...]
        shifted_gt_tokens = xt[:,1:,...]
         
        mse_loss = F.mse_loss(shifted_pred_tokens, shifted_gt_tokens, reduction="none").sum(dim=-1) # [b, t*s/g - 1]
        kl_div = outputs['kl_divergence'] # [b, t*s/g - 1]

        return mse_loss, kl_div
    
    def training_step(self, batch, batch_idx):
        # set_trace()
        mse_loss, kl_div = self.model_step(batch)
        loss_masks = torch.ones_like(mse_loss, dtype=torch.bool)
        loss_masks[:, :127] = False
        mse_loss = mse_loss * loss_masks
        kl_div = kl_div * loss_masks
        # set_trace()
        loss = mse_loss.mean() + self.cfg.experiment.beta * kl_div.mean()

        self.log('train_mse_loss', mse_loss.mean().item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_kl_div', self.cfg.experiment.beta * kl_div.mean().item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        # set_trace()
        mse_loss_strat_t = F.pad(mse_loss, (0,1)).unsqueeze(-1)
        kl_div_strat_t = F.pad(kl_div, (0,1)).unsqueeze(-1)
        mse_loss_strat_t = mse_loss_strat_t.reshape(mse_loss_strat_t.shape[0], 8, -1).mean(dim=0).mean(dim=1)
        kl_div_strat_t = kl_div_strat_t.reshape(kl_div_strat_t.shape[0], 8, -1).mean(dim=0).mean(dim=1)
        for i in range(len(mse_loss_strat_t)):
            self.log(f'mse_loss_strat_t={i}', mse_loss_strat_t[i])
            self.log(f'kl_div_strat_t={i}', kl_div_strat_t[i])
        return loss

    def validation_step(self, batch, batch_idx):
        print("Validation")
        mse_loss, kl_div = self.model_step(batch)
        loss = mse_loss.mean() + self.cfg.experiment.beta * kl_div.mean()
        self.log('val_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        
        total_preds = []
        epoch_folder_name=f"epoch_{self.current_epoch+1}"
        for i in tqdm(range(10)): #HARD CODED for now!!!
            x0 = torch.randn(10, 128, 6).unsqueeze(1).to(batch[0].dtype).to(self.device) # HARD CODED for now!!!
            preds = self.model.generate(x0, device=self.device).cpu().detach()
            preds = rearrange(preds, 'b (t s) (g d) -> b t (s g) d', s=x0.shape[2], g=self.model_cfg.llm.num_res_per_group)
            total_preds.append(preds[:,-1,...])
        # set_trace()
        total_preds = torch.cat(total_preds, dim=0).reshape(-1, 6)
        angles = torch.load(self.cfg.data.data_path)[:100]
        
        folder_name = os.path.join(self.exp_dir, epoch_folder_name)
        os.makedirs(folder_name, exist_ok=True)
        for i in range(6):
            
            _ = plt.hist(angles.reshape(-1, 6)[:,i], bins=100, range=(-np.pi, np.pi), label="GT angles", alpha=0.5)
            _ = plt.hist(total_preds.reshape(-1, 6)[:,i], bins=100, range=(-np.pi, np.pi), label="Predicted angles", alpha=0.5)
            
            plt.legend()
            
            plt.savefig(os.path.join(folder_name, f"angle_{i}_histogram.png"))
            if self.logger is not None and isinstance(self.logger, WandbLogger):
                self.logger.experiment.log({f"angle_{i}_histogram": [wandb.Image(os.path.join(folder_name, f"angle_{i}_histogram.png"))]})
            plt.close()
        return {
            'val_loss': loss
        }

    def on_predict_epoch_start(self):
        self.all_sample_trajectories = []

    def predict_step(self, batch, batch_idx):
        # set_trace()
        print(f"Inference temperature: {self.inference_temperature}")
        length, num_samples_per_length = batch
        x0 = torch.randn(num_samples_per_length.item(), length.item(), 6).unsqueeze(1).to(torch.float).to(self.device)   
        preds = self.model.generate(x0, device=self.device, temperature=self.inference_temperature).cpu().detach() # []
        preds_np= preds.numpy()
        self.all_sample_trajectories.append(preds_np)
        
    def on_predict_epoch_end(self, results):
        outdir = self.exp_dir
        self.sampled = []
        for array in self.all_sample_trajectories:
            # Iterate through the first dimension (batch dimension) of the array
            for i in range(array.shape[0]):
                self.sampled.append(array[i])
        # Now, self.sampled is a list of n_samples, where each element is an np array of shape [n_timepoints, sample_length, 6]
        # Thus satisfying the requirements of FoldingDiff code
        final_sampled = [s[-1] for s in self.sampled]
        sampled_dfs = [
            pd.DataFrame(s, columns=['phi', 'psi', 'omega', 'tau', 'CA:C:1N', 'C:1N:1CA']) # Hard coded for simplicity
            for s in final_sampled
        ]

        # Plot histograms for the angles
        total_sample_angles = np.vstack(final_sampled)
        # set_trace()
        train_angles = torch.load("/home/sh2748/foldingdiff/protein_angles_train.pt")
        reshaped_train_angles = train_angles.reshape(-1,6)
        mask = torch.any(reshaped_train_angles != 0, dim=1)
        non_zero_angles = reshaped_train_angles[mask]
        filtered_non_zero_angles = non_zero_angles[torch.randperm(non_zero_angles.size(0))][:total_sample_angles.shape[0]]
        for i in range(6):
            _ = plt.hist(filtered_non_zero_angles[:,i], bins=100, range=(-np.pi, np.pi), label="GT angles", alpha=0.5)
            _ = plt.hist(total_sample_angles[:,i], bins=100, range=(-np.pi, np.pi), label="Predicted angles", alpha=0.5)
            
            # plt.ylim(0, 14000)
            plt.legend()
            plt.savefig(os.path.join(outdir, f"angle_{i}_histogram.png"))
            if self.logger is not None and isinstance(self.logger, WandbLogger):
                self.logger.experiment.log({f"angle_{i}_histogram": [wandb.Image(os.path.join(outdir, f"angle_{i}_histogram.png"))]})
            plt.close()

        # set_trace()
        # Write the raw sampled items to csv files
        sampled_angles_folder = os.path.join(outdir, "sampled_angles")
        os.makedirs(sampled_angles_folder, exist_ok=True)
        print(f"Writing sampled angles to {sampled_angles_folder}")
        for i, s in enumerate(sampled_dfs):
            s.to_csv(os.path.join(sampled_angles_folder, f"generated_{i}.csv.gz"))
        # Write the sampled angles as pdb files
        pdb_files = self.write_preds_pdb_folder(sampled_dfs, os.path.join(outdir, "sampled_pdb"))

        # Write the angles
        full_history_angles_dir = os.path.join(sampled_angles_folder, "sample_history")
        os.makedirs(full_history_angles_dir)
        full_history_pdb_dir = os.path.join(outdir, "sampled_pdb/sample_history")
        os.makedirs(full_history_pdb_dir)
        # sampled is a list of np arrays
        for i, sampled_series in enumerate(self.sampled):
            snapshot_dfs = [
                pd.DataFrame(snapshot, columns=['phi', 'psi', 'omega', 'tau', 'CA:C:1N', 'C:1N:1CA'])
                for snapshot in sampled_series
            ]
            # Write the angles
            ith_angle_dir = os.path.join(full_history_angles_dir, f"generated_{i}")
            os.makedirs(ith_angle_dir, exist_ok=True)
            for timestep, snapshot_df in enumerate(snapshot_dfs):
                snapshot_df.to_csv(
                    os.path.join(ith_angle_dir, f"generated_{i}_timestep_{timestep}.csv.gz")
                )
            # Write the pdb files
            ith_pdb_dir = os.path.join(full_history_pdb_dir, f"generated_{i}")
            self.write_preds_pdb_folder(
                snapshot_dfs, ith_pdb_dir, basename_prefix=f"generated_{i}_timestep_"
            ) 
    
    @staticmethod
    def write_preds_pdb_folder(
        final_sampled,
        outdir: str,
        basename_prefix: str = "generated_",
        threads: int = multiprocessing.cpu_count(),
    ):
        """
        Write the predictions as pdb files in the given folder along with information regarding the
        tm_score for each prediction. Returns the list of files written.
        Copied from foldingdiff/bin/sample.py
        """
        os.makedirs(outdir, exist_ok=True)
        print(
            f"Writing sampled angles as PDB files to {outdir} using {threads} threads"
        )
        # Create the pairs of arguments
        arg_tuples = [
            (os.path.join(outdir, f"{basename_prefix}{i}.pdb"), samp)
            for i, samp in enumerate(final_sampled)
        ]
        # Write in parallel
        with multiprocessing.Pool(threads) as pool:
            files_written = pool.starmap(create_new_chain_nerf, arg_tuples)

        return files_written