import torch
import numpy as np
from typing import Union

class IntegralFlowMatcher:
    """
    Organize IFM methods in a class
    """
    def __init__(self, sigma_std: Union[float, int] = 0.0, sigma: Union[float, int] = 0.0, same_time = True, time_interp = True, noise_on_x0=True, noise_on_x1=True,):
        """Initialize the IntegralFlowMatcher class. It requires hyper-parameters sigma and sigma_std

        Parameters
        ----------
        sigma :         Union[float, int], represents the amount of noise we add to the paths
        sigma_std:      Union[float, int], represents the std of the normal distribution to sample sigma from
        same_time:       Boolean, whether all trajectories share the same set of sampled time points
        time_interp:     Boolean, whether to interpolate time points instead of random sampling
        noise_on_x0:     Boolean, whether to add noise to x0
        noise_on_x1:     Boolean, whether to add noise to x1
        
        Notes:
        ------
            Don't confuse with these parameters.
            1. sigma
                We add noise to the trajectories using Normal(0, sigma)
            2. sigma_std
                When exploring fixed point losses, we want to vary the noise added to the trajectories. 
                This is achieved by varying sigma.
                We sample sigma according to Normal(sigma, sigma_std)
                
        """
        if not (sigma_std == 0):
            print("Varying noise according to N({}, std={})".format(sigma, sigma_std))
        else:
            print("Noise is not varying.")  
        self.sigma = sigma
        self.sigma_std = sigma_std
        self.same_time = same_time
        self.time_interp = time_interp
        self.noise_on_x0 = noise_on_x0
        self.noise_on_x1 = noise_on_x1
        
    def compute_mu_t(self, x0, x1, t):
        """
            Compute the mean of the probability path that is, t * x1 + (1 - t) * x0.
            This is the unnoised trajectory, that we want to predict.

            Parameters
            ----------
            x0 : Tensor, shape [batch_size, space_dim], represents the source minibatch
            x1 : Tensor, shape [batch_size, space_dim], represents the target minibatch
            t : FloatTensor, shape [batch_size, num_per_path]

            Returns
            -------
            mu_t: t * x1 + (1 - t) * x0, shape [batch_size, sqeuence_length, space_dim]
        """
        
        batch_size = x0.shape[0]
        token_size = x0.shape[1]
        sequence_length = t.shape[1]

        # x0 and x1 have shape [batch_size, space_dim]
        x0_unsqueezed = x0.unsqueeze(1)  # shape will be [batch_size, 1, space_dim]
        x1_unsqueezed = x1.unsqueeze(1)  # shape will be [batch_size, 1, space_dim]

        x0_broadcasted = x0_unsqueezed.expand(batch_size, sequence_length, token_size)
        x1_broadcasted = x1_unsqueezed.expand(batch_size, sequence_length, token_size)
        # x0_broadcasted and x1_broadcasted have shape [batch_size, sqeuence_length, space_dim]

        t_expanded = t.unsqueeze(-1).expand(-1, -1, token_size) # shape will be [batch_size, sqeuence_length, space_dim]
        
        # element-wise, shape [batch_size, sqeuence_length, space_dim]
        mu_t = t_expanded * x1_broadcasted + (1 - t_expanded) * x0_broadcasted

        return mu_t
    
    def sample_sigma(self):
        '''
            Sample sigma from N(self.sigma, std=self.sigma_std)
            
        '''
        epsilon = torch.randn(1)
        return self.sigma + self.sigma_std * epsilon

    def sample_xt(self, x0, x1, t, noise_on_x0=False, noise_on_x1=False):
            """
            Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

            Parameters
            ----------
            x0 :             Tensor, shape [batch_size, space_dim]
                                 represents the source minibatch
            x1 :             Tensor, shape [batch_size, space_dim]
                                 represents the target minibatch
            t :              FloatTensor, shape [batch_size, num_per_path]
            

            Returns
            -------
            xt : Tensor, shape [batch_size, sqeuence_length, space_dim]
            mu_t: Tensor, shape [batch_size, sqeuence_length, space_dim], computed with compute_mu_t

            """
            sigma = self.sample_sigma().to(x0.device)
                
            mu_t = self.compute_mu_t(x0, x1, t) # shape [batch_size, sqeuence_length, space_dim]
            epsilon = torch.randn_like(mu_t) # shape [batch_size, sqeuence_length, space_dim]
            if not self.noise_on_x0:
                epsilon[:, 0, :] = 0
            if not self.noise_on_x1:
                epsilon[:, -1, :] = 0
            xt = mu_t + sigma * epsilon # shape [batch_size, sqeuence_length, space_dim]
            
            return xt, mu_t
        
    def sample_time(self, num_paths, num_per_path):
        '''
                Sample the time points

                Parameters
                ----------
                num_per_path:    Int, the number of time points to sample (uniformly) per each path
                device:          String, CPU or cuda

                Returns
                -------
                t_sorted:        Tensor, shape [batch_size, num_per_path],
                                        represents the sorted sampled (or interpolated) time points

                Note
                ----
                t_sorted will start with 0 and end with 1.
        '''
        assert num_per_path > 2

        if self.time_interp:
            if not self.same_time:
                print("Same time is enforced if doing interpolation for time.")
            t = torch.tensor(np.linspace(0,1,num_per_path)) # [1, sequence_length]
            return t.repeat(num_paths, 1)
        else:
            if self.same_time:
                random_points = torch.rand(1, num_per_path - 1)# shape [batch_size, sqeuence_length]
                sorted_points, _ = torch.sort(random_points, dim=1)
                sorted_points = sorted_points.repeat(num_paths, 1)
            else:
                random_points = torch.rand(num_paths, num_per_path - 1) # shape [batch_size, sqeuence_length]
                sorted_points, _ = torch.sort(random_points, dim=1)

        sorted_points[:, 0] = 0
        ones = torch.ones(num_paths, 1)
        t_sorted = torch.cat((sorted_points, ones), dim=1)

        return t_sorted

    def sample_conditional_flow(self, x0, x1, num_per_path,  device="cpu"):
        '''
            Sample a conditional flow based on sampled source and target points.
            
            Parameters
            ----------
            x0 :             Tensor, shape [batch_size, space_dim], represents the source minibatch
            x1 :             Tensor, shape [batch_size, space_dim], represents the target minibatch
            num_per_path:    Int, the number of time points to sample (uniformly) per each path
            device:          String, CPU or cuda
            
            Returns
            -------
            t_sorted:        Tensor, shape [batch_size, num_per_path],
                                    represents the sorted sampled (or interpolated) time points
            xt:              Tensor, shape [batch_size, num_per_path, space_dim],
                                    the sampled conditional path
            mu_t:            Tensor, shape [batch_size, num_per_path, space_dim],
                                    the ground truth conditional path
            
            Note
            ----xiang
                1. x0, x1 shape may be 4D. Need discussing.
            

                [version update 11/19]: x0 and x1 is also included in the xt, correspondingly t=0 t=1 is included in t
                [version update 11/21]: now supports x0 and x1 to be 3 or 4 dimensional, x0 and x1 is excluded from xt (cause problems)
                [version update 12/3]: if x0 is 3-dimensional, xt will be 4-dimensional
                [version update 12/10]: add same_time flag.
                    If same_time = True, we sample the same set of time points for each trajectory
                        This is required for real ANIE and optional for fake ANIE
                    If same_time = False, we sample different time points for different trajectories     
                [version update 12/17]: add x0, x1, t=0, t=1 back to xt and t
                [version update 12/26]: add time_interp flag.
                    If time_interp = True, then sample time as a linear interpolation between 0 and 1.
                        Since for every trajectory, time goes from 0 to 1, same_time flag is by default also set to True.
                    If time_interp = False, then sample time randomly. Whether all trajectories share the same set of time points then depend on the same_time flag.
                [problem as on 1/17]: 4 dimensional input may not work
        '''
        
        *batch_set_dim, token_size = x0.shape
        x0_flat = x0.reshape(-1, token_size)
        x1_flat = x1.reshape(-1, token_size)
        num_paths = x0_flat.shape[0]
        
        t_sorted = self.sample_time(num_paths, num_per_path).to(device)
        xt, mu_t = self.sample_xt(x0=x0_flat, x1=x1_flat, t=t_sorted) # shape [num_paths, sqeuence_length, space_dim]
        
        return t_sorted.to(device), xt.to(device), mu_t.to(device)