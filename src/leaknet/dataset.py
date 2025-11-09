from torch.utils.data import Dataset
from os import listdir, path
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from .utils import get_distance_from_file_name

from .config import NORMAL_DATA, ANOMALOUS_DATA, TRAIN_SIZE, VAL_SIZE, BATCH_SIZE
import lightning as L

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
                    distance = int(get_distance_from_file_name(path))
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
    """Pad sequences using last value."""
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

class LeakNetDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        
        dataset = LeakNetDataset()
        self.n_features = dataset.n_features
        
        train_size = int(TRAIN_SIZE * len(dataset))
        val_size = int(VAL_SIZE * len(dataset))
        test_size = len(dataset) - train_size - val_size

        self.train_set, self.val_set, self.test_set = random_split(dataset,
                                                                   [train_size, val_size, test_size],
                                                                   torch.Generator().manual_seed(torch.initial_seed()))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)