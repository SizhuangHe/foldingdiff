import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

# Define a PyTorch Dataset class
class MyDataset(Dataset):
    def __init__(self, dataframe=None, data_path=None):
        if data_path is not None:
            print(f"Initializing from a data path: {data_path}")
            self.data = torch.load(data_path)
        elif data_path is None and dataframe is not None:
            print("Initializing from a dataframe.")
            self.data = torch.stack(dataframe['angles'].tolist())
        else:
            raise Exception("Dataframe and data path are both None!")
        
         
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ProteinAngleDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.train_dataset, self.val_dataset = random_split(MyDataset(data_path=self.data_dir), [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


