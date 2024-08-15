# from .models import *
import copy
import os
import torch
import torch.nn as nn

def get_device():
    return torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

class Params:
    def __init__(self, **kwargs):
        self.update(**kwargs)
        
    def update(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def print(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
            
    def save_attr(self, save_to_dir, filename="parameters.pth"):
        save_to_dir = os.path.join(save_to_dir, filename)
        torch.save(self.__dict__, save_to_dir)
    
    def load_attr(self, filename):
        loaded_attr = torch.load(filename)
        self.__dict__.update(loaded_attr)
        
            
# def params2model(params):
#     if params.solver_type == "self_attn":
#         raise Exception("Not maintained for long")
#         self_attn = SelfAttention(input_dim=params.embed_dim, num_heads = params.num_heads)
#         self_attn_solver = Solver(self_attn, num_blocks=params.num_blocks, num_iterations=params.num_iterations)
#         model = FakeANIE(input_dim = params.input_dim, embed_dim=params.embed_dim, ff_dim = params.ff_dim, solver=self_attn_solver, spatial_integration=params.spatial_integration, trajectory_embedding_type = params.trajectory_embedding_type).to(params.device)
#         num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         return model, num_parameters
    
#     elif params.solver_type == "transformer":
#         transformer_layer = nn.TransformerEncoderLayer(d_model=params.embed_dim, nhead = params.num_heads, dim_feedforward=params.transf_ff_dim,batch_first=True, dropout=params.transformer_dropout)
#         transformer = nn.TransformerEncoder(transformer_layer, num_layers=params.num_blocks)
#         transformer_solver = Solver(transformer, num_blocks=1, num_iterations=params.num_iterations)
#         model = FakeANIE(input_dim = params.input_dim, embed_dim=params.embed_dim, mlp_dropout=params.mlp_dropout, ff_dim = params.ff_dim, solver=transformer_solver, spatial_integration=params.spatial_integration, trajectory_embedding_type = params.trajectory_embedding_type, volterra=params.volterra, sample_iter_type=params.sample_iter_type, sample_param=params.sample_param, smooth_fac=params.model_smooth_fac, internal_loss_type=params.internal_loss_type, max_iter=params.max_iter, skip_flag=params.skip_flag).to(params.device)
#         num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         return model, num_parameters
#     else:
#         raise Exception("Model type not supported or unspecified.")
        
        