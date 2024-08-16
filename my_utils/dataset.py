import torch
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
import pytorch_lightning as pl

class CustomDataset(Dataset):
    def __init__(self, data_path, lengths_path):
        self.data = torch.load(data_path)
        self.lengths = torch.load(lengths_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.lengths[idx]

class LengthBasedSampler(Sampler):
    def __init__(self, lengths, batch_size):
        # Make sure the length of each protein within the same is the same
        self.lengths = lengths
        self.batch_size = batch_size
        
        # Group indices by length
        self.indices_by_length = {}
        for idx, length in enumerate(lengths):
            if length.item() not in self.indices_by_length:
                self.indices_by_length[length.item()] = []
            self.indices_by_length[length.item()].append(idx)

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

class ProteinAngleDataModule(pl.LightningDataModule):
    def __init__(self, 
        train_angles_dir="protein_angles_train.pt", 
        train_lengths_dir="protein_lengths_train.pt", 
        val_angles_dir="protein_angles_val.pt",
        val_lengths_dir="protein_lengths_val.pt", 
        test_angles_dir="protein_angles_test.pt",
        test_lengths_dir="protein_lengths_test.pt", 
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
        sampler = LengthBasedSampler(torch.load(self.train_lengths_dir), self.batch_size)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, batch_sampler=sampler)
    
    def val_dataloader(self):
        sampler = LengthBasedSampler(torch.load(self.val_lengths_dir), self.batch_size)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, batch_sampler=sampler)


