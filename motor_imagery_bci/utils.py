import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)   
        return x, self.y[idx]

def create_dataloaders(X, y, batch=32, split=0.7):
    n = len(X)
    n_train = int(n * split)
    n_val = int((n - n_train) * 0.5)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]

    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]

    train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=batch, shuffle=True)
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=batch)
    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=batch)

    return train_loader, val_loader, test_loader
