from os.path import *
from torch.utils.data import Dataset, DataLoader

class GBIDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
    
def GBI_data_loader(X, y, batch_size):
    GBI_dataset = GBIDataset(X, y)
    GBI_dataloader = DataLoader(GBI_dataset, batch_size=batch_size, shuffle=False)
    
    return GBI_dataloader
