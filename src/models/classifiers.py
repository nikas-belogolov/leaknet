import torch
from torch import nn
from abc import ABC
from config import *
import torch.nn.functional as F
from models.model import BaseModel

class ClassifierModel(BaseModel, ABC):
  
  def __init__(self, lambda_distance=0.001):
    super().__init__()
    self.lambda_distance = lambda_distance
  
  def training_step(self, batch) -> torch.Tensor:
    x, y_target = batch
    y_pred = self.forward(x)
    return self._compute_loss(y_pred, y_target)
  
  def validation_step(self, batch) -> torch.Tensor:
    x, y_target = batch
    y_pred = self.forward(x)
    return self._compute_loss(y_pred, y_target)
  
  def _compute_loss(self, pred, target):
    pred_label, pred_dist = pred
    true_label, true_dist = target
    
    anomaly_mask = true_label == 1
    
    loss_distance = anomaly_mask * F.mse_loss(pred_dist, true_dist) # calculate distance loss only for true anomalies
    loss_anomaly = F.binary_cross_entropy(pred_label, true_label)
    
    # add distance loss only for true anomalies
    
    return loss_anomaly + self.lambda_distance * loss_distance
  
  def configure_optimizers(self):
    self.optimizer = torch.optim.Adam(self.parameters(), lr=self._hparams["learning_rate"])
    self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)


class RNNClassifier(ClassifierModel):
    def __init__(self, input_dim, hidden_dim=4, num_layers=1, dropout=0, learning_rate=0, **kwargs):
      super().__init__(**kwargs)
      
      self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=num_layers)
      
      self.dropout = nn.Dropout(dropout)
      self.sigmoid = nn.Sigmoid()
      
      self.anomaly_head = nn.Linear(hidden_dim, 1)
      self.distance_head = nn.Linear(hidden_dim, 1)
      nn.init.xavier_uniform_(self.anomaly_head.weight)
      nn.init.xavier_uniform_(self.distance_head.weight)

    def forward(self, x) -> torch.Tensor:
      _, latent = self.rnn(x)
      latent = latent[-1]
      anomaly = self.sigmoid(self.dropout(self.anomaly_head(latent)))
      distance = self.dropout(self.distance_head(latent))
      return anomaly.squeeze(), distance.squeeze()
    
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