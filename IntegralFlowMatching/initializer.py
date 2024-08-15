import torch
from .utils import get_device

device = get_device()

class Initializer: # assume xt is 3-dim for now
    '''
        This class is used to initialize xt. 
        [Tested on Nov 26] works as expected but initializing with multiple time points is slow due to the for-loops
        [update on Jan 17] supports random masking for multi-dimensional trajectories, but main still contain some bugs. Will fix them when those functionalities are needed
    '''
    def __init__(self, init_method, num_points, device):
        self.init_method = init_method
        self.num_points = num_points
        self.device = device
        
    def _sample_indices(self, batch_size, sequence_length):
        """
        Sample num_init unique indices for each sequence in the batch,
        ensuring that 0 is not included, and then sort them.
        """
#         indices_sampled = torch.zeros(batch_size, self.num_points, dtype=torch.long)
#         for i in range(batch_size):
#             indices = torch.randperm(sequence_length - 1)[:self.num_points] + 1
#             indices_sampled[i] = indices - 1
#         sorted_indices = torch.sort(indices_sampled, dim=1).values
#         print("sampled indices: ", sorted_indices)
#         return sorted_indices

        # Sample indices within the range [1, sequence_length - 2], excluding the start and end, ensuring no repeats
        sorted_indices = []
        for _ in range(batch_size):
            indices = torch.randperm(sequence_length - 2)[:self.num_points] + 1  # Sample 3 unique indices
            indices = indices.sort().values
            sorted_indices.append(indices)
        sorted_indices = torch.stack(sorted_indices)
        return sorted_indices
    
    def _initialize_xt_with_noise(self, xt, mean=0.0, std=1.0):
        noise = torch.normal(mean, std, size=xt.shape).to(self.device)
        return xt + noise
    
    def _initialize_x0(self, xt, x0):
        if len(x0.shape) == 2:
            x0 = x0.unsqueeze(1)
            x0 = x0.repeat(1, xt.size(1), 1) 
        elif len(x0.shape) == 3:
            x0 = x0.unsqueeze(2)
            x0 = x0.repeat(1, 1, xt.size(2), 1) 
        
        return x0
    
    def _initialize_random(self, xt):
        if len(xt.shape) == 3:
            batch_size, sequence_length, space_dim = xt.shape
            # Randomly select indices for each path
            random_indices = torch.randint(sequence_length, (batch_size, 1)).unsqueeze(-1).expand(-1, -1, space_dim).to(self.device)
            # shape, [batch_size, 1, space_dim], the space_dim dimension contains the index
            random_tokens = torch.gather(xt, 1, random_indices) 
            # shape, [batch_size, 1, space_dim], the space_dim dimension contains the actual token
            xt_initialized = random_tokens.expand(-1, sequence_length, -1)
            # shape, [batch_size, sequence_length, space_dim], populate the entire path with the chosen token
            return xt_initialized
        elif len(xt.shape) == 4:
            batch_size, set_size, sequence_length, space_dim = xt.shape
            # Randomly select indices for each sequence in each set for each batch
            random_indices = torch.randint(sequence_length, (batch_size, set_size, 1, 1)).to(self.device)
            # Gather the random tokens
            random_tokens = torch.gather(xt, 2, random_indices.expand(-1, -1, sequence_length, space_dim))
            # Expand the selected tokens across each sequence
            xt_initialized = random_tokens.expand(-1, -1, sequence_length, -1)
            return xt_initialized
        
    def _initialize_mask_noise(self, xt, mean=0.0, std=1.0, mask_token=None):
        if len(xt.shape) == 4:
            raise Exception("May contain bugs")
            batch_size, set_size, sequence_length, token_size = xt.shape

            # Randomly select indices for each sequence
            indices = torch.rand(batch_size, set_size, sequence_length).argsort(dim=-1)[:, :, :self.num_points] # may contain bugs here
            
            # Create a mask for selected tokens
            mask = torch.zeros(batch_size, set_size, sequence_length, 1, dtype=torch.bool)
            mask.scatter_(2, indices.unsqueeze(-1), 1)

            # Expand the mask to cover the token size dimension
            mask = mask.expand(-1, -1, -1, token_size).to(self.device)

            # Generate Gaussian noise
            
            if mask_token is None:
                noise = torch.normal(mean, std, size=xt.shape).to(self.device)
                xt[~mask] = noise[~mask]
            else:
                xt[~mask] = mask_token

            return xt

        elif len(xt.shape) == 3:
            batch_size, sequence_length, token_size = xt.shape
            mask_token = torch.full((token_size,), mask_token, device=xt.device)

            # select the indices to keep
            random_indices = torch.rand(batch_size, sequence_length)
            _, selected_indices = torch.sort(random_indices, descending=True)
            selected_indices = selected_indices[:, :self.num_points].to(self.device)
            
                # Initialize a mask filled with False
            mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=xt.device)
                # Use scatter_ to set selected indices to True
            mask.scatter_(1, selected_indices, True)
            mask = mask.unsqueeze(-1).expand(-1, -1, token_size)

               # Generate Gaussian noise
            if mask_token is None:
                raise NotImplementedError
                noise = torch.normal(mean, std, size=xt.shape).to(self.device)
                xt = torch.where(mask, xt, noise)
            else:
                mask_token_expanded = mask_token.unsqueeze(0).unsqueeze(0).expand(batch_size, sequence_length, -1).to(self.device)  # Expand mask_token to match xt shape
                xt = torch.where(mask, xt, mask_token_expanded)
            return xt, selected_indices   
        else:
            raise NotImplementedError
            
    def _initialize_with_first_two(self, xt):
        # Assuming tensor shape is [batch_size, sequence_length, 1]
        batch_size, sequence_length, _ = xt.shape
        assert sequence_length > 1

        # Creating a mask of the same shape, initialized to False
        mask = torch.zeros(batch_size, sequence_length, 1, dtype=torch.bool).to(self.device)
        # Setting the first two tokens to True
        mask[:, 0:2] = True

        return xt * mask
    
    def _initialize_step(self, xt, x0):                  
        batch_size, sequence_length, _ = xt.shape
        sorted_indices = self._sample_indices(batch_size, sequence_length)
        initialized_tensor = torch.zeros_like(xt)
        for i in range(batch_size):
            indices = sorted_indices[i]
            for j in range(sequence_length):
                # think of valid_indices as the steps we have access to, we pick the largest one to actually step onto
                valid_indices = indices[indices <= j]
                index = valid_indices.max().item() if valid_indices.numel() > 0 else 0
                initialized_tensor[i, j] = x0[i] if index == 0 else xt[i, index]
        return initialized_tensor
    
    def _initialize_linear_interpolation(self, xt, x0, x1):
        """
            Initialize a new tensor using linear interpolation between the tokens at the sampled indices.
            We first sample a number of tokens (except the first and the last) in each sequence of xt, 
            fix them, put x0 at the first place and x1 at the last. Finally, interpolate to fill all unfilled places.
        """
        batch_size, sequence_length, token_size = xt.shape
        sorted_indices = self._sample_indices(batch_size, sequence_length)
        initialized_tensor = torch.zeros_like(xt)

        for i in range(batch_size):
            indices = sorted_indices[i]
            initialized_tensor[i, 0] = x0[i]  # Start with x0
            initialized_tensor[i, -1] = x1[i]  # End with x1
            for idx in indices:
                initialized_tensor[i, idx] = xt[i, idx]  # Place sampled tokens
            

            prev_idx = 0
            prev_token = x0[i]

            for idx in indices.tolist() + [sequence_length - 1]:
                current_token = initialized_tensor[i, idx]
                steps = idx - prev_idx

                for k in range(1, steps):
                    t = k / steps
                    interpolated_token = (1 - t) * prev_token + t * current_token
                    initialized_tensor[i, prev_idx + k] = interpolated_token

                prev_idx = idx
                prev_token = current_token

        return initialized_tensor
    
    def initialize(self, xt, x0, x1, std=1.0, mask_token=None):
        '''
        Intialize xt to feed into our solver
        
        Parameters
        ----------
        xt: Tensor, shape [batch_size, sequence_length, space_dim]
            represents sampled conditional paths (of sequence_length)
        init_method: str
            None:           we don't do any initialization
            "xt_noise":     we add some Gaussian noise to xt
            "x0":           initialize with x0
            "random":       for each path, we randomly select one token (among the sequence_length tokens)
                and initialize all tokens of that path to be that token
            "step": initialize with multiple tokens, as a step function. 
                e.g.: if intialize with 2 time points t1 and t2: 
                    tokens before t1 initialized to x0, 
                    between t1 and t2 initialized to x_t1,
                    after t2 initialized to x_t2
            "interpolation":     initialize with a linear interpolation
                e.g.: if intialize with 2 time points t1 and t2:
                    tokens before t1 initialized by interpolating x0 and x_t1, 
                    between t1 and t2 initialized by interpolating x_t1 and x_t2
                    after t2 initialized by x_t2 and x1
            "mask_noise":        we sample a number of tokens, fix them and 
                                    either: 1. replace the rest with Gaussian(0,1) noise
                                            2. replace the rest with mask_token
            "initialize_with_first_two": replace all but the first two tokens
        num_init: int, number of time points to initialize with
                specify a number of time points and the actually time points are randomly sampled
                If using "random", this will be omitted because "random" only initialize with one
            
    '''
        if self.init_method is None:
            return xt.to(self.device), None
        elif self.init_method == "xt_noise":
            return self._initialize_xt_with_noise(xt), None
        elif self.init_method == "x0":
            return self._initialize_x0(xt, x0).to(self.device), None
        elif self.init_method == "random":
            return self._initialize_random(xt).to(self.device), None
        elif self.init_method == "step":
            return self._initialize_step(xt, x0).to(self.device), None
        elif self.init_method == "interpolation":
            return self._initialize_linear_interpolation(xt, x0, x1).to(self.device), None
        elif self.init_method == "mask_noise":
            xt, indices = self._initialize_mask_noise(xt, std=std, mask_token=mask_token)
            return xt.to(self.device), indices
        elif self.init_method == "initialize_with_first_two":
            return self._initialize_with_first_two(xt), None
        else:
            raise NotImplementedError