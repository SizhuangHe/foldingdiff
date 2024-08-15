import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import copy
import numpy as np

class EmbedMLP(nn.Module):
    '''
        This is a simple MLP that projects spatial coordinates or time points to a higher dimensional space.
    '''
    def __init__(self, 
                 input_dim=2,
                 hidden_dim=512,
                 output_dim=16,
                 p_dropout = 0):
        
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=p_dropout)
    
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class SelfAttention(nn.Module):
    '''
        This module simply set Q,K,V to be the same
        so that we don't have to specify them when calling nn.MultiheadAttention.
    '''
    def __init__(self, 
                 input_dim,
                 num_heads, 
                 dropout=0.0, 
                 bias=True, 
                 add_bias_kv=False, 
                 add_zero_attn=False, 
                 kdim=None, 
                 vdim=None, 
                 batch_first=True, 
                 device=None, 
                 dtype=None):
        super().__init__()
        self.module = MultiheadAttention(input_dim, num_heads, dropout, bias, 
                 add_bias_kv, add_zero_attn, kdim, vdim, 
                 batch_first, device, dtype)
    
    def forward(self, x, need_weights=True):
        return self.module(query=x, key=x, value=x, need_weights=need_weights)[0]   
    
class Solver(nn.Module):
    '''
        Iterating of the solver is deprecated.
    '''
    def __init__(self, solver_block: nn.Module, num_blocks, num_iterations):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_iterations = num_iterations
        self.solver_block = solver_block
    
    def _next_x(self, old_x, new_x, smooth_fac):
        return smooth_fac * old_x + (1 - smooth_fac) * new_x
    
    def forward(self, x, mask):
        x = self.solver_block(x, mask=mask)
        return x
    
class FakeANIE(nn.Module):
    '''
        This is a fake ANIE. 
        It simply project the spatial coordinates and the time step to a high dimensional space separately,
        add them up to form tokens and feed the tokens to the self attention.
    '''
    def __init__(self,
                 input_dim,
                 embed_dim,
                 ff_dim,
                 solver: Solver,
                 mlp_dropout=0.0, 
                 spatial_integration = False,
                 trajectory_embedding_type=None,
                 volterra=False,
                 sample_iter_type = None,
                 sample_param = 1,
                 max_iter = 10,
                 smooth_fac = 0,
                 internal_loss_type = None,
                 skip_flag = True
                 ):
        '''
        Prameters
        ---------
            input_dim:                  (int), the spatial dimension
            embed_dim:                  (int), the latent dimension
            ff_dim:                     (int), the hidden dimension of MLPs
            solver:                     (an instance of the Solver class), is typically a Transformer Encoder
            mlp_dropout:                (float), the drop out ratio of MLPs
            spatial_integration:        (boolean), flag on whether or not doing spatial integration
            trajectory_embedding_type:  (string or None), see Notes.
            volterra:                   (boolean), whether we are using a Volterra or Fredholm equation
            sample_iter_type:           (string or None), how we sample the number of iterations
            sample_param:               (float), the parameter of the distribution we sample from
            smooth_fac:                 (float), the smoothing factor of the iterations
            internal_loss_type:         (float or None), the type of internal loss
            skip_flag:                       (boolean), whether use skip connection at each iteration
        
        Notes:
        ---------
            1. trajectory_embedding_type:
                if trajectory_embedding_type is None:
                    x0 is not embedding to xt
                if trajectory_embedding_type is "add"
                    x0, xt and t are embedded to a latent space of dimension embed_dim and added together
                if trajectory_embedding_type is "cat"
                    x0 and xt are embedded embedded to dimension embed_dim/2 and concatenated
                    t is embedded to embed_dim
                    they are then added
            2. volterra:
                if True, Volterra; if False, Fredholm
                implementation-wise:
                    if False, no masking is applied. Every token attends to every other token.
                    if True, a causal mask is applied. The tokens only attends to tokens ahead of them.
                        The mask looks like: (for example: sequence length is 3)
                            [[0   , -inf, -inf]
                             [0   , 0   , -inf]
                             [0   , 0   ,    0]]
            3. sample_iter_type:
                if sample_iter_type is "exp":
                    we sample the number of iterations according to exponential(sample_param),
                        the corresponding pdf (when sample_rate=1) is f(x)=exp(-x)
                        we need to add a 1 to the sampled value to make it meaningful (we don't want 0 iterations)
                        the sampled iteration should not excceed max_iter
                if sample_iter_type is "uni", 
                    for sample_param of the time, we do 1 iteration
                    for 1- sample_param of the time, we sample uniformly from 2 to max_iter
                if sample_iter_type is "fixed"
                    we do a fixed number of iterations
                    max_iter is the number of iterations
                if sample_iter_type is None,
                    we don't do any iteration
                    If we don't do iteration, smoothing factor is automatically zero
            4. sample_param:
                if sample_iter_type is "exp":
                    sample_param is the rate lambda
                if sample_iter_type is "uni":
                    sample_param is the success rate of the first bernoulli trial
                sample_param is designed so that larger sample_param will produce a distribution more concentrated at the origin
            5. max_iter:
                if max_iter = -1, there is no limitation on the number of iterations
            6. internal_loss_type:
                if internal_loss_type is None, then no internal_loss is computed
                if internal_loss_type is "fixed_pt_vanilla", the sum of ||x_{k+1}-x_k|| is computed
                if internal_loss_type is "fixed_pt_last", after the all iterations, ||f(out)-out|| is computed
                    if not doing any iterations aat all, this is equivalent as fixed_pt_vanilla
                if internal_loss_type is "fixed_pt_unnoised", we predict on noised trajectory and compute loss
                    this is implemented in the training script
        
        '''
        super().__init__()
        
        self.timeMLP = EmbedMLP(input_dim = 1, hidden_dim = ff_dim, output_dim = embed_dim, p_dropout=mlp_dropout)
        
        self.solver = solver
        self.decoder = nn.Linear(in_features=embed_dim, out_features = input_dim)
        self.spatial_integration = spatial_integration
        self.sample_param = sample_param
        self.sample_iter_type = sample_iter_type
        self.max_iter = max_iter
        self.smooth_fac= smooth_fac
        self.volterra = volterra
        self.trajectory_embedding_type = trajectory_embedding_type
        self.internal_loss_type = internal_loss_type
        self.skip_flag = skip_flag
        
        if self.trajectory_embedding_type is not None:
            if self.trajectory_embedding_type == "add":
                print("Embedding x0 to trajectories by addition!")
                embed_dim = embed_dim
            elif self.trajectory_embedding_type == "cat":
                print("Embedding x0 to trajectories by concatenation!")
                embed_dim = int(embed_dim/2)
            else:
                 raise NotImplementedError
            self.trajectory_embed = EmbedMLP(input_dim = input_dim, hidden_dim = ff_dim, output_dim = embed_dim, p_dropout=mlp_dropout)
        
        self.spaceMLP = EmbedMLP(input_dim = input_dim, hidden_dim = ff_dim, output_dim = embed_dim, p_dropout=mlp_dropout)
        
        if self.sample_iter_type in ["exp", "uni"]:
            if self.smooth_fac == 0:
                raise Exception("Smoothing factor cannot be zero while requiring iterations!")
            else:
                print("Requiring iterations. The smoothing factor is {}".format(self.smooth_fac))
            if self.sample_iter_type=="exp":
                print("Sampling the number of iterations from Exponential({}) and the max number of iteration is {}".format(self.sample_param, self.max_iter))
            elif self.sample_iter_type=="uni":
                print("Sampling the number of iterations is a two-step process.")
                print("First, a Bernoulli trial to whether to do iteration or not. (probability to NOT do iteration is: {})".format(self.sample_param))
                print("Then, uniformaly sample from [2, {}] is the Bernoulli trial results in a 1.".format(self.max_iter))
            elif self.sample_iter_type == "fixed":
                if self.max_iter < 1:
                    raise Exception("Number of iteration should be at least 1.")
                elif self.max_iter == 1:
                    print("Warning: Doing 1 iteration, equivalent to no iteration!")
                    self.sample_iter_type = None
                else:
                    print("Doing a fixed number of {} iterations.".format(self.max_iter))
                 # this is for the sanity of _sample_niter method
        else:
            print("Not requiring iterations, smoothing factor is NOT set to zero! The smoothing factor is {}".format(self.smooth_fac))
            
        
        if self.volterra:
            print("Modeling with a Volterra Integral Equation.")
        else:
            print("Modeling with a Fredholm Integral Equation.")
    
    def _sample_niter(self):
        '''
            This function samples an integer according to Exponential(self.sample_rate) distribution. 
            This integer is used as the number of iterations.
            
            We sample a float, take its integer part and add 1 to the sampled result to make it meaningful as the number of iterations.
        
        '''
        if self.sample_iter_type == "exp":
            rate = self.sample_param
            scale = 1/rate
            sample = np.random.exponential(scale)
            num_iter = int(sample) + 1
        elif self.sample_iter_type == "uni":
            p = self.sample_param
            bernoulli_outcome = np.random.binomial(n=1, p=p)
            if bernoulli_outcome == 1:
                num_iter = 1
            else:
                assert self.max_iter > 0
                num_iter = np.random.randint(low=2, high=self.max_iter+1)
        elif self.sample_iter_type == "fixed":
            num_iter = self.max_iter 
        elif self.sample_iter_type is None:
            num_iter = 1
        else:
            raise Exception("Unsupported type of sampling method!")
        if self.max_iter > 0:
            num_iter = min(num_iter, self.max_iter)
                   
        return num_iter
    
    def _mask(self, sequence_length):
        '''
            This function creates the mask used for different kinds of IEs.
            If it's a Volterra equation, a causal mask is the output.
            If it's a Fredholm equation, a trivial mask (all 0's) is the output.
        '''
        if self.volterra:
            mask = (torch.triu(torch.ones(sequence_length, sequence_length)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        else:
            mask = None
        return mask
        
    def _infer_shape(self, x):
        '''
            First, infer the type of integration based on the shape of x and self.spatial_integration. 
            Then reshape x into corresponding shapes.
            
            Valid input shape is:
                - 3-dimensional, assumed to be [batch_size, sequence_length, token_size]. 
                    In this case, it's impossible to integrate over space.
                    No matter what self.spatial_integration is, only temporal integration is to be performed.
                    No reshape needed.
                    
                - 4-dimensional, [batch_size, set_size, sequence_length, token_size]
                    In this case, we may or may not want to integrate over space but it's possible.
                    If self.spatial_integration is True, 
                        reshape into [batch_size, set_size*sequence_length, token_size]
                    If not, 
                        reshape into [batch_size*set_size, sequence_length, token_size]
        '''
            
        if len(x.shape) == 3:
            return x
        elif len(x.shape) == 4:
            batch_size, set_size, sequence_length, token_size = x.shape
            
            if not self.spatial_integration:
                return x.reshape(-1, sequence_length, token_size)
            else:
                return x.reshape(batch_size, -1, token_size)
        else:
            raise Exception("Invalid input shape. Must be 3 dimensional or 4 dimensional")
    
    @staticmethod
    def _smooth_connect(smooth_fac, old_tokens, new_tokens):
        '''
            This function implements the smooth connection between iterations.
            The smooth connection is computed as below:
                y_{k+1} = alpha * y_k + (1 - alpha) * y_{k+1}
        '''
        return old_tokens * smooth_fac + (1 - smooth_fac) * new_tokens

    def _embed_identifier(self, x0, xt_embeddings):
        '''
            This function embeds x0 to xt based on self.trajectory_embedding_type.
        '''
        if self.trajectory_embedding_type is not None:
            x0_expanded = x0.unsqueeze(len(x0.shape)-1).expand_as(xt)
            trajectory_embeddings = self.trajectory_embed(x0_expanded)
            if self.trajectory_embedding_type == "add":
                space_embeddings = xt_embeddings + trajectory_embeddings
            elif self.trajectory_embedding_type == "cat":
                space_embeddings = torch.cat((xt_embeddings, trajectory_embeddings), dim=-1)
            else:
                raise NotImplementedError
        else:
            space_embeddings = xt_embeddings
         
        return space_embeddings
            
    def _solve(self, x0, xt, f, t):
        '''
            This function implements one complete pass through the model. 
            There is a bit of an abuse of notation here when we call it "solve". Maybe we should clarify what should be called a solver, whether the entire model or only the transformer inside.
        
            Parameter
            ---------
            x0:     Tensor, shape [batch_size, space_dim]
                        the source data points
            xt:     Tensor, shape [batch_size, sequence_length, space_dim]
                        the current guess of the trajectory. This usually comes from the initialization or the previous iteration.
            f:      Tensor, shape [batch_size, sequence_length, space_dim]
                        the free function, usually the initialization
            t:      Tensor, shape [batch_size, sequence_length, 1]
                        the time points of each token
        '''
        xt_embeddings = self.spaceMLP(xt)
        time_embeddings = self.timeMLP(t)
        
        space_embeddings = self._embed_identifier(x0, xt_embeddings)
        time_embeddings = time_embeddings.reshape(space_embeddings.shape)
        
        
        trajectories = space_embeddings + time_embeddings
        
        trajectories = self._infer_shape(trajectories)
        _, sequence_length, _= trajectories.shape
        mask = self._mask(sequence_length)
        trajectories = self.solver(trajectories, mask=mask)
        trajectories = self.decoder(trajectories)

        if self.skip_flag:
            trajectories += f # skip connection
        
        return trajectories
          
    def forward(self, x0, xt, t, inf_init=None, n_iter=None):
        '''
            The forward function.

            Parameter
            ---------
            x0:     Tensor, shape [batch_size, space_dim]
                        the source data points
            xt:     Tensor, shape [batch_size, sequence_length, space_dim]
                        the initialization of the trajectories
            t:      Tensor, shape [batch_size, sequence_length, 1]
                        the time points
            n_iter: Int or None
                        this provides a handle to override the number of iterations when needed
            inf_init:   Tensor or None
                        this provides a handle to pass in the initialization during inference
                        This is due to a different way to iteration during inference.
                        In inference, xt refers to the solved trajectory from the previous iteration and init is the initialized trajectory.
                        In training, xt is the initialized trajectory.

            Returns
            -------
            tensor
                    solved trajectories
            int
                    the actual number of iterations sampled/fixed
            tensor
                    internal loss
        '''
        num_iter = self._sample_niter()
        if n_iter is not None: # allow overriding the number of iterations
            num_iter = n_iter
        
        internal_loss = torch.tensor(0, dtype=torch.float).to("cuda")
        
        if inf_init is not None:
            # Override initialization only in evaluation mode and when passing in an initialization
            initialization = inf_init
        else:
            initialization = xt
 
        trajectories = xt
        
        for i in range(num_iter):
            old_trajectories = trajectories
            new_trajectories = self._solve(x0, old_trajectories, initialization, t)
            trajectories = self._smooth_connect(self.smooth_fac, old_trajectories, new_trajectories)
            if self.internal_loss_type == "fixed_pt_vanilla":
                internal_loss += torch.dist(old_trajectories, trajectories, p=2)
        
        if self.internal_loss_type == "fixed_pt_last": 
            new_traj = self._solve(x0, trajectories, initialization, t)
            internal_loss += torch.dist(new_traj, trajectories, p=2)
                
        
        return trajectories, num_iter, internal_loss