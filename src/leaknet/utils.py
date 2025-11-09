import random
import numpy as np
import torch
import os
from .config import *
import re

def listdir_abs(path: str):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            paths.append(os.path.abspath(os.path.join(root, file)))
    return paths

def get_distance_from_file_name(filename: str):
    return re.search('\d+(?=m)', filename).group()

def set_device() -> torch.device:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.set_default_device(device)
    print(f"Using device = {torch.get_default_device()}")
    return device

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Using seed = {seed}")
    return seed