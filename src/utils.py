import random
import numpy as np
import torch
import os
from data.dataset import LeakAnomalyDetectionDataset
from config import *

def listdir_abs(path: str):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            paths.append(os.path.abspath(os.path.join(root, file)))
    return paths

def is_interactive():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False
    
def set_device() -> torch.device:
    # Check if CUDA is available
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    torch.set_default_device(device)
    print(f"Using device = {torch.get_default_device()}")
    
    return device
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(obj: object, name: str):
    torch.save(obj, name + ".pth")
    
def load_model(path: str) -> torch.nn.Module:
    return torch.load(path, weights_only=False)