import random
import numpy as np
import torch
import os
from .config import *
from sklearn.metrics import precision_recall_curve
import importlib
import matplotlib.pyplot as plt
import re

def listdir_abs(path: str):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            paths.append(os.path.abspath(os.path.join(root, file)))
    return paths

def get_distance_from_file_name(filename: str):
    return re.search('\d+(?=m)', filename).group()

def load_model_from_trial(trial):
    model = getattr(importlib.import_module(f"models"), trial.user_attrs["model_name"])
    model = model(**trial.user_attrs["hparams"])
    model_state_dict = torch.load(trial.user_attrs["model_path"], weights_only=True)['model_state_dict']
    model.load_state_dict(model_state_dict)
    return model

def get_best_theshold(y_true: np.array, y_pred: np.array):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold

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