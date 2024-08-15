import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from .datasets import *
from .optimal_transport import *

            
class InferencePlotter:
    def __init__(self, dim):
        self.dim = dim
        
    def _inference_plot_1D(self, pred, x0, x1, time_steps):
        # Trajectory plot
        #print("t:{}, trajectories:{}".format(t.shape, trajectories.shape))
        plt.figure(figsize=(10, 6))
        colors = cm.viridis(pred.cpu()[:,0,:])
        for i in range(pred.shape[0]):
            plt.plot(time_steps.cpu()[i, :, 0],pred.cpu()[i, :, 0],  marker='', linestyle='-', linewidth=0.3, color=colors[i])
        plt.colorbar(cm.ScalarMappable(cmap='viridis'), orientation='vertical')
        plt.xlim(-0.125, 1.125)  
        plt.show()
        plt.close()

        # Heat map plot of learned trajectories
        H, xedges, yedges = np.histogram2d(pred.cpu().reshape(-1), time_steps.reshape(-1).cpu(), bins=100)
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(yedges, xedges, H, shading='auto')  # Transpose H to align axes
        plt.colorbar(label='Frequency')
        plt.title('Heatmap of (x, t) Occurrences')
        plt.xlabel('Time (t)')
        plt.ylabel('Position (xt)')
        plt.show()
        plt.close()

        # Model output distribution
        plt.figure(figsize=(10, 6))
        min_value = min(x1.reshape(-1).numpy().min(), pred[:,-1,:].cpu().reshape(-1).min())
        max_value = max(x1.reshape(-1).numpy().max(), pred[:,-1,:].cpu().reshape(-1).max())
        bin_edges = np.linspace(min_value, max_value, 16)
        plt.hist(x1.reshape(-1).numpy(), bins=bin_edges, alpha=0.5, color="salmon", edgecolor="black", label="target")
        plt.hist(pred[:,-1,:].cpu().reshape(-1), bins=bin_edges, alpha=0.5, color="skyblue", edgecolor="black", label="learned target")
        plt.legend()
        plt.title("Model output distribution")
        plt.show()
        plt.close()
        
        # Histogram of the source and the start of the trajectory
            # trying to understand how the model change the source distribution
        plt.figure(figsize=(10, 6))
        plt.hist(x0.reshape(-1).numpy(), bins=bin_edges, alpha=0.5, color="salmon", edgecolor="black", label="Source")
        plt.hist(pred[:,0,:].cpu().reshape(-1), bins=bin_edges, alpha=0.5, color="skyblue", edgecolor="black", label="Start of the trajectory")
        plt.legend()
        plt.title("Comparison of source and start of the (learned) trajectory")
        plt.show()
        plt.close()
    
    def _inference_plot_2D(self, pred, x0, x1):
        # Learned Trajectory plot
        plt.figure(figsize=(8, 8))
        plt.axis('equal')
#         plt.xlim([-8, 8])
#         plt.ylim([-8, 8])
        for i in range(x0.shape[0]):
            plt.scatter(x0[i,0], x0[i,1], s = 1, color='green')   
        plt.scatter(x1[:, 0], x1[:, 1], s = 1, color="orange")
        
#         trajectories = pred.reshape(-1, num_per_flow, 2).detach().cpu().numpy()
        for i in range(pred.shape[0]):
            plt.plot(pred.cpu()[i, :, 0], pred.cpu()[i, :, 1], marker='', linestyle='-', linewidth=0.1, color='gray')
            # Mark starting point in red
            plt.plot(pred.cpu()[i, 0, 0], pred.cpu()[i, 0, 1], marker='o', markersize=1, color='red', label='Start' if i == 0 else "")
            # Mark ending point in blue
            plt.plot(pred.cpu()[i, -1, 0], pred.cpu()[i, -1, 1], marker='o', markersize=1, color='blue', label='End' if i == 0 else "")
        plt.title("Inference trajectories")
        plt.show()
        plt.close()
        
        # Heatmap of the predicted target
        print(pred.shape)
        heatmap, xedges, yedges = np.histogram2d(pred.cpu().numpy()[:, -1, 0], pred.cpu().numpy()[:, -1, 1], bins=(64, 64))
        plt.figure(figsize=(8, 6))
        plt.axis('equal')
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', aspect='auto')
        plt.colorbar()
        plt.title("Heatmap of the predicted target")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()
        plt.close()
        
    def plot(self, pred, x0, x1, time_steps):
        if self.dim == 1:
            return self._inference_plot_1D(pred, x0, x1, time_steps)
        elif self.dim == 2:
            return self._inference_plot_2D(pred, x0, x1)
        else:
            # print("No plots are being made because the dimension is {}".format(self.dim))
            return
            

class IFMinferer:
    def __init__(self, inference_type, source_dataset, target_dataset, batch_size, num_per_flow, device, init_method, perturb_std=0,mean=0, std=0, std_inc_rate=0, dim=2, mask_token=0):
        '''
            Initialize an IFM inferer
            
            Parameters
            ----------
            inference_type:      (string)
            experiment_name:     (string), the name of the experiment. 
                                        Choose from the following three:
                                            1. "1G_2G": [1-dimensional] 1 Gaussian to 2 Gaussians
                                            2. "1G_3G": [1-dimensional] 1 Gaussian to 3 Gaussians
                                            3. "8G_2M": [2-dimensional] 8 Gaussians to 2 Moons
            batch_size:          (int), the amount of data points (from the source distribution) generated for inference
            num_per_flow:        (int), number of time points per flow, these are generated by linear interpolating 0 and 1
            device:              (torch.device), cpu or cuda
            init_method:         (string), method to initialize a trajectory
                                        supports "noise" and "mask"
            mean, std:           (float), the mean and std of the Gaussian noise applied to x0
                                            Note: don't confuse with mean and std in self.masking. 
                                                Those refer to the noise of the noise mask (although in practice the 0 token is used as a mask instead of noise).
            std_inc_rate.   (float), the rate of increasing the std of the Gaussian noise along the sequence_length dimension
                                            For example: if std=sigma, sequence is l and std_inc_rate is alpha
                                                the stds of each token is then:
                                                    [sigma, sigma+alpha, sigma+2*alpha, ..., sigma+(l-1)alpha
                                            This is defaulted to zero (same std for noise for each token)
        '''
        self.inference_type = inference_type
        self.source_dataset = IFMdatasets(batch_size=batch_size, dataset_name = source_dataset, dim=dim, gaussian_var=1)
        self.target_dataset = IFMdatasets(batch_size=batch_size, dataset_name = target_dataset, dim=dim, checker_size=4, gaussian_var=0.1)
        self.plotter = InferencePlotter(dim=dim)
        self.batch_size = batch_size
        self.num_per_flow = num_per_flow
        self.device = device
        self.init_method = init_method
        self.mean = mean
        self.std = std
        self.std_inc_rate = std_inc_rate
        self.mask_token = mask_token
        self.perturb_std = perturb_std
        print("Initialize trajectories with {}".format(self.init_method))
        assert self.init_method in ["noise", "mask", "noise_mask", "perturb_mask"]
        if self.init_method in ["mask", "noise_mask"]:
            print("The masking token is {}".format(self.mask_token))
        
    def _expand_to_sequence(self, x0):
        return x0.unsqueeze(1).repeat(1, self.num_per_flow, 1)

    def _init_trajectories(self, x0):
        if self.init_method == "noise":
            trajectories = self._add_noise_to(x0)
        elif self.init_method == "mask":
            trajectories = self.masking(x = x0.to(self.device))
        elif self.init_method == "noise_mask":
            trajectories = self._add_noise_to(x0)
            trajectories = self.masking(x = trajectories.to(self.device))
        elif self.init_method == "perturb_mask":
            trajectories = self.masking(x = x0.to(self.device))
        else:
            raise Exception("Unsupported initialization method!")
        return trajectories

    
    def _basic_inference(self, x0, model, prev_trajectory, initialization, plot):
        time_steps = torch.tensor(np.linspace(0, 1, self.num_per_flow), dtype=torch.float32).repeat(self.batch_size, 1).unsqueeze(-1)
        with torch.no_grad():  
            model.eval()
            if initialization is not None:
                initialization = initialization.to(self.device)
            pred, num_iter,_ = model(x0=x0.to(self.device), xt=prev_trajectory.to(self.device), t=time_steps.to(self.device), inf_init=initialization)

        x1 = self.target_dataset.generate_data()
        if plot:
            self.plotter.plot(pred, x0[:,0,:], x1, time_steps)                
        
        mmd = rbf_mmd(pred[:, -1,:], x1, sigma_list=[0.01, 0.1, 1, 10, 100])
        wasserstein_1 = wasserstein(pred[:, -1,:], x1, p=1)
        wasserstein_2 = wasserstein(pred[:, -1,:], x1, p=2)
        distance_dict = {
            "rbf-MMD": mmd,
            "1-Wasserstein": wasserstein_1,
            "2-Wasserstein": wasserstein_2
        }

        prev_new_traj_mmd = rbf_mmd(pred[:, -1,:], prev_trajectory[:, -1, :], sigma_list=[0.01, 0.1, 1, 10, 100])
        prev_new_traj_wasserstein_1 = wasserstein(pred[:,-1,:], prev_trajectory[:, -1, :], p=1)
        prev_new_traj_wasserstein_2 = wasserstein(pred[:,-1,:], prev_trajectory[:, -1, :], p=2)
        dist_prev_new_traj_dict = {
            "rbf-MMD": prev_new_traj_mmd,
            "1-Wasserstein": prev_new_traj_wasserstein_1,
            "2-Wasserstein": prev_new_traj_wasserstein_2
        }


        return pred, distance_dict, dist_prev_new_traj_dict
    
    def masking(self, x): 
        batch_size, sequence_length, token_size = x.shape
        mask_token = torch.full((token_size,), self.mask_token, device=x.device, dtype=x.dtype)  # Specify dtype

        # Create a mask that is False for the first token and True for the rest
        mask = torch.ones(batch_size, sequence_length, dtype=torch.bool, device=x.device)
        mask[:, 0] = False  # Keep the first token unmasked
        mask[:, 1] = False  # Keep the second token unmasked
        mask = mask.unsqueeze(-1).expand(-1, -1, token_size)

        if mask_token is None:
                raise NotImplementedError("Gaussian noise masking not supported!")
        else:
            # Expand mask_token to match x shape
            mask_token_expanded = mask_token.unsqueeze(0).unsqueeze(0).expand(batch_size, sequence_length, -1).to(self.device)
            x = torch.where(mask, mask_token_expanded, x)  # Apply mask, the first argument is the condition tensor, the second is the value to keep if true, the third is the value to use if false
        if self.init_method == "perturb_mask":
            # add noise to the second token
            noise = torch.normal(0, self.perturb_std, size=(batch_size, 1, token_size), device=x.device)
            x[:, 1] += noise.squeeze(1)
        return x
    
    def _add_noise_to(self, x):
        batch_size, sequence_length, _ = x.shape

        # Create a scaling factor that increases along the sequence_length dimension
        scaling_factor = torch.arange(sequence_length) * self.std_inc_rate + self.std

        # Normal(0,1) noise
        noise = torch.randn_like(x)

        # Apply the scaling factor to the noise (unsqueeze to match the dimensions)
        noise *= scaling_factor.unsqueeze(0).unsqueeze(-1)

        return x + noise + self.mean
    
    def _one_step_inference(self, model, plot):
        dist_dicts = []
        dist_prev_new_dicts = []
        x0 = self.source_dataset.generate_data()
        x0 = self._expand_to_sequence(x0)
        trajectories = self._init_trajectories(x0)
        
        trajectories, dist_dict, dist_prev_new_dict = self._basic_inference(x0, model, trajectories, plot)
        dist_dicts.append(dist_dict)
        dist_prev_new_dicts.append(dist_prev_new_dict)
        return trajectories, dist_dicts, dist_prev_new_dicts
    
    def _iterative_inference(self, model, num_iterations: int, plot, plot_last):
        assert num_iterations > 0
        dist_dicts = []
        dist_prev_new_dicts = []
        x0 = self.source_dataset.generate_data()
        x0 = self._expand_to_sequence(x0)
        initialization = self._init_trajectories(x0)
        trajectories = initialization
        
        for i in tqdm(range(num_iterations)):
            if i == num_iterations - 1 and plot_last:
                trajectories, dist_dict, dist_prev_new_dict = self._basic_inference(x0=x0, model=model, prev_trajectory=trajectories, initialization=initialization, plot=True)
            else:
                trajectories, dist_dict, dist_prev_new_dict = self._basic_inference(x0=x0, model=model, prev_trajectory=trajectories, initialization=initialization, plot=plot)
            
            dist_dicts.append(dist_dict)
            dist_prev_new_dicts.append(dist_prev_new_dict)
        return trajectories, dist_dicts, dist_prev_new_dicts
    
    def inference(self, model, num_iterations, plot=True, plot_last = True):
        if self.inference_type == "one_step":
            return self._one_step_inference(model, plot=plot)
        elif self.inference_type == "iterative":
            return self._iterative_inference(model, num_iterations, plot=plot, plot_last=plot_last)
        else:
            raise NotImplementedError
         