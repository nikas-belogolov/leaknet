import torch
from torch import nn
from abc import ABC
from config import *

from models.model import BaseModel

class ClassifierModel(BaseModel, ABC):
  
  def __init__(self):
    super().__init__()
    self._criterion = nn.BCELoss()
  
  def training_step(self, batch) -> torch.Tensor:
    x, y = batch
    y_pred = self.forward(x)
    loss = self._criterion(y_pred, y)
    return loss
  
  def validation_step(self, batch) -> torch.Tensor:
    x, y = batch
    y_pred = self.forward(x)
    val_loss = self._criterion(y_pred, y)
    return val_loss
  
  def configure_optimizers(self):
    self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
    self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)


class RNNClassfier(ClassifierModel):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0):
      super().__init__()
      
      self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=num_layers)
      
      self.fc = nn.Linear(hidden_dim, 1)
      nn.init.xavier_uniform_(self.fc.weight)
      
      self.dropout = nn.Dropout(dropout)
      self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
      _, output = self.rnn(x)
      output = self.dropout(self.fc(output[0]))
      output = self.sigmoid(output)
      return output.squeeze()
    
class CNNRNNClassifier(RNNClassfier):
  """
  CNNRNNClassifier: extract features before passing to rnn
  """
  
  def __init__(self, input_dim, hidden_dim, cnn_filters=1, num_layers=1, dropout=0):
    super().__init__(cnn_filters, hidden_dim, num_layers, dropout)
    
    # CNN 1d feature extractor
    self.conv = nn.Sequential(
      nn.Conv1d(input_dim, cnn_filters, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool1d(2, 2)
    )
    
  def forward(self, x) -> torch.Tensor:
    
    x = x.permute(0, 2, 1)
    x = self.conv(x)
    x = x.permute(0, 2, 1)
    
    return super().forward(x)