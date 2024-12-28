import torch
import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from config import *

# def cross_val(dataset, n_splits):
#     kf = KFold(n_splits)
#     for i, (train_index, val_index) in enumerate(kf.split(dataset)):
        
#         train_set = Subset(dataset, train_index)
#         val_set = Subset(dataset, val_index)
        
#         train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
#         val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
        
        # Epoch iterations here