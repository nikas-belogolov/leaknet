from torch.utils.data import Dataset
from os import listdir, path
import pandas as pd
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import re

import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
import utils

from config import NORMAL_DATA, ANOMALOUS_DATA, TRAIN_SIZE, VAL_SIZE, BATCH_SIZE

class LeakNetDataset(Dataset):
    
    def __init__(self, with_delta=False, with_distance=False):
        super().__init__()
        
        self.data = []
        self.ids = []
        self.labels = []
        
        self.with_delta = with_delta
        self.with_distance = with_distance
        
        for files_dir, label in zip([NORMAL_DATA, ANOMALOUS_DATA], range(2)):
            for file in listdir(files_dir):
                df = self._read_csv(path.join(files_dir, file))
                self.data.append(df)

                if self.with_distance:
                    distance = int(utils.get_distance_from_file_name(path))
                    self.labels.append((label, distance)) # Anomaly and distance label
                else:
                    self.labels.append(label)

        
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        
        self.n_features = self.data[0].size(-1)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
    def _read_csv(self, path):
        df = pd.read_csv(path)
        df = df.set_index('timestamp')
        df = df.sort_index()
        
        # Feature engineering
        if self.with_delta:
            df["delta"] = np.abs(df["pressure"] - df["flow"])
            df = df[["delta", "pressure", "flow"]]
        else:
            df = df[["pressure", "flow"]]
        
        df = df.to_numpy(dtype=np.float32)
        df = torch.tensor(df, dtype=torch.float32)
        
        return df
    

def pad_sequence_replicate(sequences):
    max_len = max([seq.size(0) for seq in sequences])
    
    padded_sequences = []
    
    for s in sequences:
        pad_amount = max_len - s.size(0)
        s = F.pad(s.unsqueeze(0), (0, 0, 0, pad_amount), "replicate")
        padded_sequences.append(s)

    return torch.cat(padded_sequences)
    

def collate_fn(batch):
    data, labels = zip(*batch)
    data = pad_sequence_replicate(data)
    labels = torch.stack(labels)
    return data, labels

def get_datasets(dataset):
    train_size = int(TRAIN_SIZE * len(dataset))
    val_size = int(VAL_SIZE * len(dataset))
    test_size = len(dataset) - train_size - val_size

    return random_split(dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(torch.initial_seed()))

def get_dataloader(dataset, shuffle, collate_fn):
    return DataLoader(dataset, BATCH_SIZE, shuffle=shuffle, collate_fn=collate_fn)

def get_dataloaders(train_set, val_set, test_set, collate_fn):
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader