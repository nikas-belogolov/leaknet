import torch
from torch import nn
from models.model import BaseModel
from config import *

# class EnsembleModel(BaseModel):
    
#     def __init__(self, models):
#         super().__init__()
        
#         self.models = nn.ModuleList(models)
        
    
#     def training_step(self, batch) -> torch.Tensor:
#         x, y = batch
#         y_pred = self.forward(x)
#         loss = self._criterion(y_pred, y)
#         return loss
    
#     def validation_step(self, batch) -> torch.Tensor:
#         x, y = batch
#         y_pred = self.forward(x)
#         val_loss = self._criterion(y_pred, y)
#         return val_loss
    
#     def configure_optimizers(self):
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
#         self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)