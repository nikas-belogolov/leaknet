from torchvision import transforms, utils
from torchvision.transforms import Normalize
import torch
import numpy as np
import pandas as pd

class Normalize(torch.nn.Module):
  def __init__(self, mean, std):
    super().__init__()
    self.mean = mean
    self.std = std
    
  def __call__(self, tensor: torch.Tensor):
      return (tensor - self.mean) / self.std