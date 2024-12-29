import torch
from torch import nn
from abc import ABC
from config import *

from models.model import Model

class Classifier(Model, ABC):
  
  _criterion = nn.BCELoss()

  def __init__(self):
    super().__init__()
  
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
  
class RNNClassfier(Classifier):
    def __init__(self, input_dim, hidden_dim, rnn_unit=nn.GRU, num_layers=1, dropout=0):
      super().__init__()
      
      self.rnn = rnn_unit(input_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=num_layers)
      self.fc = nn.Linear(hidden_dim, 1)
      self.dropout = nn.Dropout(dropout)
      self.sigmoid = nn.Sigmoid()
      nn.init.xavier_uniform_(self.fc.weight)
      nn.init.zeros_(self.fc.bias)
      

    def forward(self, x) -> torch.Tensor:
      _, output = self.rnn(x)
      output = self.dropout(self.fc(output[0]))
      output = self.sigmoid(output)
      return output.squeeze()
    
    
    
class CNNRNNClassifier(Classifier):
  def __init__(self, input_dim, hidden_dim, rnn_unit=nn.GRU, num_layers=1, dropout=0):
    super(CNNRNNClassifier, self).__init__()
    
    # CNN 1d feature extractor
    self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=3)
    self.relu = nn.ReLU()
    nn.init.kaiming_normal_(self.conv.weight)
    
    # RNN 
    self.rnn = rnn_unit(input_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=num_layers)
    
    
    self.fc = nn.Linear(hidden_dim, 1)
    nn.init.xavier_uniform_(self.fc.weight)
    self.dropout = nn.Dropout(dropout)
    self.sigmoid = nn.Sigmoid()
     
  def forward(self, x) -> torch.Tensor:
    
    output = self.relu(self.conv(x))
    
    _, output = self.rnn(output)
    output = self.dropout(self.fc(output[0]))
    output = self.sigmoid(output)
    return output