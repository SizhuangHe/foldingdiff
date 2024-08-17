import torch
from torch.utils.data import Dataset, DataLoader, random_split, Sampler, DistributedSampler
import pytorch_lightning as pl
from ipdb import set_trace
import math
import random
import numpy as np

class LengthBasedSampler(Sampler):
    def __init__(self, lengths, batch_size, drop_last=False):
        # Make sure the length of each protein within the same is the same
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by length
        self.indices_by_length = {}
        for idx, length in enumerate(lengths):
            if isinstance(length, torch.Tensor):
                length = length.item()
            if length not in self.indices_by_length:
                self.indices_by_length[length] = []
            self.indices_by_length[length].append(idx)

        # Create a list of batches where all sequences in each batch have the same length
        self.batches = []
        for length, indices in self.indices_by_length.items():
            for i in range(0, len(indices), batch_size):
                self.batches.append(indices[i:i + batch_size])
        
    def __iter__(self):
        # Shuffle the batches for randomness
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)

class DistributedLengthBasedSampler(DistributedSampler):
    def __init__(self, lengths, batch_size, drop_last=False, num_replicas=None, rank=None, shuffle=True, seed=123):
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0
        self.shuffle = shuffle
        self.seed = seed
        self.np_rng = np.random.default_rng(self.seed)

        print(f"Sampler initialized with rank={rank}, num_replicas={num_replicas}")
        
        # Group indices by length
        self.indices_by_length = {}
        for idx, length in enumerate(lengths):
            if isinstance(length, torch.Tensor):
                length = length.item()
            if length not in self.indices_by_length:
                self.indices_by_length[length] = []
            self.indices_by_length[length].append(idx)

        self._create_batches()
        

    def _create_batches(self):
         # Create batches where all sequences in each batch have the same length
        self.batches = []
        for length, indices in self.indices_by_length.items():
            for i in range(0, len(indices), self.batch_size):
                self.batches.append(indices[i:i + self.batch_size])
        
        # Number of batches for each replica
        self.total_batches = len(self.batches)
        self.num_batches_per_replica = math.ceil(self.total_batches / self.num_replicas)
    
    def _shuffle_within_groups(self):
        shuffled_indices_by_length = {}
        for length, indices in self.indices_by_length.items():
            if self.shuffle:
                indices = self.np_rng.permutation(indices).tolist()
            shuffled_indices_by_length[length] = indices
        self.indices_by_length = shuffled_indices_by_length
    
    
    def __iter__(self):
        self._shuffle_within_groups() # shuffle the data within each lengths group for each epoch to form new batches
        self._create_batches()
        # Calculate the indices for the current replica
        start = self.rank * self.num_batches_per_replica
        end = min(start + self.num_batches_per_replica, self.total_batches)
        replica_batches = self.batches[start:end]

        # Shuffle batches if necessary
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.rank)  # Ensures that each replica gets a different shuffle
        if self.rank == 0:
            replica_batches = torch.randperm(len(replica_batches), generator=rng).tolist()
        else:
            replica_batches = list(range(len(replica_batches)))
        
        assert np.unique([len(torch.unique(self.lengths[self.batches[i]])) for i in replica_batches]) == 1, "Variable lengths in one batch detected!"
        # Return the batches for this replica
        return iter([self.batches[i] for i in replica_batches])
    
    def __len__(self):
        return self.num_batches_per_replica