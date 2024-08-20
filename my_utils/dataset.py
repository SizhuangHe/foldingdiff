import torch
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
import pytorch_lightning as pl
from ipdb import set_trace
import numpy as np
from .custom_sampler import DistributedLengthBasedSampler, LengthBasedSampler

class CustomDataset(Dataset):
    def __init__(self, data_path, lengths_path):
        self.data = torch.load(data_path)
        # TODO: should shuffle here!
        self.lengths = torch.load(lengths_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.lengths[idx]

class PredictionLengthDataset(Dataset):
    def __init__(self, num_samples_per_length=10, min_length=60, max_length=128):
        self.lengths = np.arange(min_length, max_length)
        self.num_samples_per_length = num_samples_per_length
    
    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        return self.lengths[idx], self.num_samples_per_length


class ProteinAngleDataModule(pl.LightningDataModule):
    def __init__(self, 
        train_angles_dir="/home/sh2748/foldingdiff/protein_angles_train.pt", 
        train_lengths_dir="/home/sh2748/foldingdiff/protein_lengths_train.pt", 
        val_angles_dir="/home/sh2748/foldingdiff/protein_angles_val.pt",
        val_lengths_dir="/home/sh2748/foldingdiff/protein_lengths_val.pt", 
        test_angles_dir="/home/sh2748/foldingdiff/protein_angles_test.pt",
        test_lengths_dir="/home/sh2748/foldingdiff/protein_lengths_test.pt", 
        batch_size: int = 32):
        super().__init__()
        self.train_angles_dir = train_angles_dir
        self.train_lengths_dir = train_lengths_dir
        self.val_angles_dir = val_angles_dir
        self.val_lengths_dir = val_lengths_dir
        self.test_angles_dir = test_angles_dir
        self.test_lengths_dir = test_lengths_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.train_dataset = CustomDataset(self.train_angles_dir, self.train_lengths_dir)
        self.val_dataset = CustomDataset(self.val_angles_dir, self.val_lengths_dir)
        self.test_dataset = CustomDataset(self.test_angles_dir, self.test_lengths_dir)

    def train_dataloader(self):
        rank = self.trainer.local_rank
        if isinstance(self.trainer.gpus, list):
            num_replicas = self.trainer.num_nodes * len(self.trainer.gpus)
        elif isinstance(self.trainer.gpus, int):
            num_replicas = self.trainer.num_nodes * self.trainer.gpus
        else:
            raise Exception("What is self.trainer.gpus?")
        print(f"rank: {rank}, num_replicas: {num_replicas}")
        # set_trace()
        sampler = DistributedLengthBasedSampler(torch.load(self.train_lengths_dir), self.batch_size, drop_last=False, rank=rank, num_replicas=num_replicas)
        return DataLoader(self.train_dataset, batch_sampler=sampler)
        # return DataLoader(self.train_dataset, batch_size=32)
    
    def val_dataloader(self):
        sampler = DistributedLengthBasedSampler(torch.load(self.val_lengths_dir), self.batch_size)
        return DataLoader(self.val_dataset, batch_sampler=sampler)
        # return DataLoader(self.val_dataset, batch_size=32)
    
    def predict_dataloader(self, num_samples_per_length, min_length, max_length):
        predict_dataset = PredictionLengthDataset(num_samples_per_length=num_samples_per_length, min_length=min_length, max_length=max_length)
        return DataLoader(predict_dataset, batch_size=1)


