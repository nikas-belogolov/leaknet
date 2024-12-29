import torch
from torch import nn
from abc import ABC
from config import *
from models.model import Model

class Autoencoder(Model, ABC):
  _criterion = nn.MSELoss()

  def __init__(self):
    super().__init__()
  
  def training_step(self, batch) -> torch.Tensor:
    x, _ = batch
    x_reconstructed = self.forward(x)
    loss = self._criterion(x_reconstructed, x)
    return loss
  
  def validation_step(self, batch) -> torch.Tensor:
    x, _ = batch
    x_reconstructed = self.forward(x)
    val_loss = self._criterion(x_reconstructed, x)
    return val_loss
  
  def configure_optimizers(self):
    self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
    self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

class SimpleAutoencoder(nn.Module):
  """
  SimpleAutoencoder: Final hidden state of the encoder is repeated and fed as input to the decoder.
  """

  def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0):
    super(SimpleAutoencoder, self).__init__()
    self.input_dim = input_dim    
    self.hidden_dim = hidden_dim
    
    self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)    
    self.decoder = nn.GRU(hidden_dim, input_dim, num_layers, batch_first=True, dropout=dropout)
    
  def forward(self, x):
    seq_len = x.size(dim=1)
    
    _, h = self.encoder(x)
    
    h_repeated = h.repeat(seq_len, 1, 1).permute(1, 0, 2)
    
    reconstructed, _ = self.decoder(h_repeated)

    return reconstructed

class ScheduledSamplingAutoencoder(nn.Module):
  """
  ScheduledSamplingAutoencoder: Decoder is either fed the previous true token or generated token. 
  """
  
  def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0):
    super(ScheduledSamplingAutoencoder, self).__init__()
    
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    
    self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
    self.decoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
    self.output_layer = nn.Linear(hidden_dim, input_dim)
    
    # Layer normalization for both encoder and decoder
    self.encoder_layer_norm = nn.LayerNorm(hidden_dim)
    self.decoder_layer_norm = nn.LayerNorm(hidden_dim)
    
  def forward(self, x, teacher_forcing_ratio=0.5):
    batch_size, seq_len, _ = x.size()
    
    _, h = self.encoder(x)
    
    # h = self.encoder_layer_norm(h)
    
    decoder_input = torch.zeros(batch_size, 1, self.input_dim)
    decoder_hidden = h
    outputs = []
    
    for t in range(seq_len):
      
      output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
      # decoder_hidden = self.decoder_layer_norm(decoder_hidden)
      output = self.output_layer(output)
      outputs.append(output)
      
      use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
      decoder_input = x[:, t, :].unsqueeze(1) if use_teacher_forcing else output
      
    return torch.cat(outputs, dim=1)



class RNNAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_unit=nn.GRU, num_layers=1):
      super(RNNAutoencoder, self).__init__()

      self.input_dim = input_dim
      self.hidden_dim = hidden_dim

      self.encoder = rnn_unit(input_dim, hidden_dim, batch_first=True)
      
      
      
      self.decoder = rnn_unit(hidden_dim, input_dim, batch_first=True)
      # self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, teacher_forcing_ratio=0.5):
      batch_size, seq_len, input_dim = x.size()
      _, hidden = self.encoder(x)

      decoder_input = x[:, 0, :].unsqueeze(1)
      outputs = torch.zeros(batch_size, seq_len, self.input_dim)

      for t in range(seq_len):
        print(decoder_input.shape, hidden.shape)
        output, hidden = self.decoder(decoder_input, hidden)

        print(output.shape, outputs.shape)

        outputs[:, t, :] = output.squeeze(1)

        if torch.rand(1).item() < teacher_forcing_ratio:
          decoder_input = x[:, t, :].unsqueeze(1)
        else:
          decoder_input = output

      return outputs