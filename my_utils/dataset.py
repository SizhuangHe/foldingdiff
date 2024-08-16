import torch
from torch.utils.data import Dataset

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