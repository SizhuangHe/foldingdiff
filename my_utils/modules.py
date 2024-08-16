import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions import Normal, kl_divergence
from .utils import modulo_with_wrapped_range

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