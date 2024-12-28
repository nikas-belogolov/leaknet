import torch
from torch import nn

criterion = nn.BCELoss()

class RNNClassfier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rnn_unit=nn.GRU, num_layers=1, dropout=0):
      super(RNNClassfier, self).__init__()
      
      self.rnn_unit = rnn_unit
      self.rnn = rnn_unit(input_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=num_layers)
      self.fc = nn.Linear(hidden_dim, output_dim)
      self.dropout = nn.Dropout(dropout)
      self.sigmoid = nn.Sigmoid()
      nn.init.xavier_uniform_(self.fc.weight)
      nn.init.zeros_(self.fc.bias)

    def forward(self, x) -> torch.Tensor:
      _, output = self.rnn(x)
      output = self.dropout(self.fc(output[0]))
      output = self.sigmoid(output)
      return output