from module import BaseModule
from torch import Tensor, optim, nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Literal
from sklearn.metrics import precision_recall_curve

class RecurrentEncoder(nn.Module):
  def __init__(self, n_features, latent_dim, rnn: nn.RNNBase):
    super().__init__()
    self.rnn = rnn(n_features, 256, bidirectional=True, batch_first=True)
    self.fc = nn.Linear(2 * 256, latent_dim)
    
  def forward(self, x):
    _, hidden = self.rnn(x)
    hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
    latent = self.fc(hidden)
    return latent

class RecurrentDecoder(nn.Module):
  def __init__(self, latent_dim, n_features, rnn_cell: nn.RNNCellBase):
    super().__init__()
    self.n_features = n_features
    self.rnn_cell = rnn_cell(n_features, latent_dim)
    self.dense = nn.Linear(latent_dim, n_features)
    
  def forward(self, x: Tensor, h_0, seq_len: int, epsilon: float=0.0):
      
    output = torch.empty_like(x)
    
    h_t = h_0.squeeze()
    # Reconstruct first element with encoder output
    x_t = self.dense(h_t)
    
    for t in range(seq_len):
      
      h_t = self.rnn_cell(x_t, h_t)
      
      # Scheduled Sampling
      use_teacher_forcing = torch.rand(1).item() < epsilon
      x_t = x[:, t, :] if use_teacher_forcing else self.dense(h_t)
      
      output[:, t, :] = self.dense(h_t)
    
    return output.view(-1, seq_len, self.n_features)
    
class RecurrentDecoderLSTM(nn.Module):
    """Recurrent decoder LSTM"""

    def __init__(self, latent_dim, n_features, rnn_cell: nn.RNNCellBase):
        super().__init__()

        self.n_features = n_features
        self.rnn_cell = rnn_cell(n_features, latent_dim)
        self.dense = nn.Linear(latent_dim, n_features)

    def forward(self, x: Tensor, h_0: Tuple[Tensor, Tensor], seq_len: int, epsilon: float=0.0):
        # Initialize output
        output = torch.empty_like(x)

        # Squeezing
        h_t: Tuple[Tensor, Tensor] = (h_0[0].squeeze(0), h_0[1].squeeze(0))

        # Reconstruct first element with encoder output
        x_t = self.dense(h_t[0])
        

        for t in range(seq_len):
      
            h_t = self.rnn_cell(x_t, h_t)
            
            # Scheduled Sampling
            use_teacher_forcing = torch.rand(1).item() < epsilon
            x_t = x[:, t, :] if use_teacher_forcing else self.dense(h_t[1])
      
            output[:, t, :] = self.dense(h_t[1])

        return output.view(-1, seq_len, self.n_features)


# Define the type hint
RNNType = Literal["GRU", "LSTM"]

class LeakNetRAE(BaseModule):
    # def __init__(self, input_size, hidden_size=64, num_layers=2, **kwargs):
    #     super().__init__()
    #     self.save_hyperparameters()
    #     self.input_size = input_size
    #     self.num_layers = num_layers
    #     self.hidden_size = hidden_size
    #     # rnn, rnn_cell = self.get_rnn_type(rnn_type)
    #     # self.decoder = self.get_decoder(rnn_type)
        
    #     # self.encoder = RecurrentEncoder(n_features, latent_dim, rnn)
    #     # self.decoder = self.decoder(latent_dim, n_features, rnn_cell)
    #     self.epsilon = 1.0 # Teacher Forcing Probability
        
    #     # For test time reconstruction error threshold calculation
    #     self.test_errors = []
    #     self.test_labels = []

    def __init__(self, input_size, hidden_size=128, num_layers=2, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.test_errors = []

        # Encoder with CNN-LSTM architecture
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//2)
        )
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size//2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # Decoder with attention mechanism
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size*2,
            hidden_size=hidden_size*2,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        # self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=4)
        self.decoder_cnn = nn.Sequential(
            nn.Conv1d(hidden_size*2, input_size, kernel_size=3, padding=1),
            # nn.Sigmoid()
        )

        self.fc = nn.Linear(hidden_size*2, hidden_size*2)
        self.layer_norm = nn.LayerNorm(hidden_size*2)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # CNN encoder
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.encoder_cnn(x_cnn).permute(0, 2, 1)

        # LSTM encoder
        enc_out, (hidden, cell) = self.encoder_lstm(x_cnn)
        hidden = self._process_bidirectional_hidden(hidden)
        cell = self._process_bidirectional_hidden(cell)

        # Decoder with attention
        decoder_input = torch.zeros(batch_size, seq_len, self.hidden_size*2, device=x.device)
        decoder_out, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        
        # Attention mechanism
        # decoder_out = decoder_out.permute(1, 0, 2)
        # attn_out, _ = self.attention(decoder_out, enc_out.permute(1, 0, 2), enc_out.permute(1, 0, 2))
        # attn_out = attn_out.permute(1, 0, 2)
        
        # CNN decoder
        decoder_out = decoder_out.permute(0, 2, 1)
        reconstructed = self.decoder_cnn(decoder_out).permute(0, 2, 1)

        return reconstructed

    def _process_bidirectional_hidden(self, h):
        h = h.view(self.num_layers, 2, -1, self.hidden_size)
        h = torch.cat([h[:, 0], h[:, 1]], dim=2).contiguous()
        return h
        
    # def forward(self, x, epsilon):
        
    #     batch_size, seq_len, _ = x.shape

    #     # Encode sequence
    #     _, (hidden, cell) = self.encoder(x)
        
    #     # Process bidirectional hidden states
    #     hidden = self._process_bidirectional_hidden(hidden)
    #     cell = self._process_bidirectional_hidden(cell)

    #     # Generate decoder input (zeros)
    #     decoder_input = torch.zeros(batch_size, seq_len, self.input_size, device=x.device)

    #     # Decode sequence
    #     decoder_out, _ = self.decoder(decoder_input, (hidden, cell))
    #     decoder_out = self.layer_norm(decoder_out)
    #     reconstructed = self.fc(decoder_out)

    #     return reconstructed

        
        # seq_len = x.shape[1]
        
        # h_n = self.encoder(x)
        # out = self.decoder(x, h_n, seq_len, epsilon)
        # return self.fc(x)
        
    def _process_bidirectional_hidden(self, h):
        # Reshape and combine bidirectional hidden states
        h = h.view(self.num_layers, 2, -1, self.hidden_size)
        h = torch.cat([h[:, 0, :, :], h[:, 1, :, :]], dim=2).contiguous()
        return h
    
    def on_fit_start(self):
        self.epsilon = 1.0
        
    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        reconstructed = self.forward(x)
        return F.mse_loss(x, reconstructed)

    def on_train_epoch_end(self):
        
        if self.trainer.current_epoch < 0: # warmup period of 10 epochs
            return
        
        self.epsilon = max(0.0, 1 - self.trainer.current_epoch / self.trainer.max_epochs)
    
    # Validation
    def validation_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        
        reconstructed = self.forward(x) # rely only on own tokens
        return F.mse_loss(x, reconstructed)
    
    # Test
    def on_test_start(self):
        pass
        # self.test_errors.clear()
        # self.test_labels.clear()
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        reconstructed = self.forward(x)
        
        # print(F.mse_loss(x, reconstructed, reduction="none").sum(dim=(1, 2)))
        
        # self.x = []
        # self.y = []
        # for x, out in zip(ZeroDivisionError, reconstructed):
        #     self.x.append(x)
        #     self.y.append(out)
        
        return reconstructed

    def on_test_end(self):
        # precision, recall, thresholds = precision_recall_curve(self.test_labels, self.test_errors)
        # f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        # best_idx = np.argmax(f1_scores)
        # self.save_hyperparameters(threshold=thresholds[best_idx])
        pass

    def predict_step(self, batch):
        reconstructed = self.forward(batch)
        print(F.mse_loss(batch, reconstructed, reduction="none").sum(dim=(1, 2)).max())

        

        # labels = F.mse_loss(batch, reconstructed, reduction="none").mean(dim=(1, 2)) > self.hparams["threshold"]
        return reconstructed
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.hparams['learning_rate'],
                    steps_per_epoch=len(self.trainer.train_loader), 
                    epochs=self.trainer.max_epochs
                ),
                'interval': 'step'
            }
        }

    @staticmethod
    def get_rnn_type(rnn_type: RNNType):
        rnn = getattr(nn, rnn_type)
        rnn_cell = getattr(nn, rnn_type + 'Cell')
        return rnn, rnn_cell

    @staticmethod
    def get_decoder(rnn_type: RNNType):
        if rnn_type == 'LSTM':
            decoder = RecurrentDecoderLSTM
        else:
            decoder = RecurrentDecoder
        return decoder
        
        
        
        
        
        
        
        

# class LeakNetRAE(BaseModule):
#     def __init__(self, n_features, latent_dim, rnn_type, **kwargs):
#         super().__init__()
#         self.save_hyperparameters()
        
#         self.test_errors = []
#         self.test_labels = []
        
#         rnn, rnn_cell = self.get_rnn_type(rnn_type)
#         self.decoder = self.get_decoder(rnn_type)
        
#         self.encoder = RecurrentEncoder(n_features, latent_dim, rnn)
#         self.decoder = self.decoder(latent_dim, n_features, rnn_cell)
#         self.epsilon = 1.0 # Teacher Forcing Probability

#     def on_fit_start(self):
#         self.epsilon = 1.0
        
#     def training_step(self, batch, batch_idx) -> Tensor:
#         x, (y_label, _) = batch
#         pred = self.forward(x, self.epsilon)
#         return self._compute_loss(pred, [x, y_label])

#     def on_train_epoch_end(self):
        
#         if self.trainer.current_epoch < 10: # warmup period of 10 epochs
#             return
        
#         self.epsilon = max(0.1, 1 - self.trainer.current_epoch / self.trainer.max_epochs)
    
#     # Validation
#     def validation_step(self, batch, batch_idx) -> Tensor:
#         x, (y_label, _) = batch
#         pred = self.forward(x, 0.0)
#         return self._compute_loss(pred, [x, y_label])
    
#     # Test
#     def on_test_start(self):
#         self.test_errors.clear()
#         self.test_labels.clear()
        
#     def test_step(self, batch, batch_idx):
#         x, (y_anomaly, y_distance) = batch
#         reconstructed, distances = self.forward(x, 0.0)
#         for x, output, label in zip(x, reconstructed, y_anomaly):
#             self.test_errors.append(
#                 self._compute_reconstruction_error(x, output).detach().cpu().numpy()
#             )
#             self.test_labels.append(label.detach().cpu().numpy())
#         return reconstructed

#     def _compute_reconstruction_error(self, x, reconstructed):
#         return F.mse_loss(x, reconstructed, reduction='mean')

#     def on_test_end(self):
#         precision, recall, thresholds = precision_recall_curve(self.test_labels, self.test_errors)
#         f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
#         best_idx = np.argmax(f1_scores)
#         self.save_hyperparameters(threshold=thresholds[best_idx])

#     def predict_step(self, batch, batch_idx):
#         x, (y_label, y_distance) = batch
#         reconstructed, distances = self.forward(x, 0.0)
#         labels = F.mse_loss(x, reconstructed, reduction='none').mean(dim=(1, 2)) >= self.hparams["threshold"]
#         return labels, distances

    
#     def forward(self, x, epsilon):
#         seq_len = x.shape[1]
#         h_n = self.encoder(x)
#         out = self.decoder(x, h_n, seq_len, epsilon)
#         return out
    
#     def _compute_loss(self, x, reconstructed):
#         return F.mse_loss(x, reconstructed)
    
#     def configure_optimizers(self):
#         self.optimizer = optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
#         self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

#     @staticmethod
#     def get_rnn_type(rnn_type: RNNType):
#         rnn = getattr(nn, rnn_type)
#         rnn_cell = getattr(nn, rnn_type + 'Cell')
#         return rnn, rnn_cell

#     @staticmethod
#     def get_decoder(rnn_type: RNNType):
#         if rnn_type == 'LSTM':
#             decoder = RecurrentDecoderLSTM
#         else:
#             decoder = RecurrentDecoder
#         return decoder
        
    
    

# class ScheduledSamplingAE(BaseModel):
#   """Decoder starts with encoder hidden state, and is either fed the previous true token or generated token."""
  
#   def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0, lambda_distance=0.001, **kwargs):
#     super().__init__()
#     self.save_hyperparameters()
    
#     self.lambda_distance = lambda_distance
#     self.test_errors = []
#     self.test_labels = []
    
#     self.input_dim = input_dim
#     self.hidden_dim = hidden_dim
    
#     self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout)
#     self.decoder = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout)
#     self.output_layer = nn.Linear(hidden_dim, input_dim)
#     self.distance_head = nn.Linear(hidden_dim, 1)
    
#     self.encoder_layer_norm = nn.LayerNorm(hidden_dim)
#     self.decoder_layer_norm = nn.LayerNorm(hidden_dim)
    
#     self.epsilon = 1.0 # Teacher Forcing Probability
    
#   def on_fit_start(self):
#     self.epsilon = 1.0
    
#   # Train
#   def training_step(self, batch, batch_idx) -> Tensor:
#     x, (y_label, y_distance) = batch
#     pred = self.forward(x)
#     return self._compute_loss(pred, [x, y_label, y_distance])

#   def on_train_epoch_end(self):
#     self.epsilon = max(0.0, 1 - self.trainer.current_epoch / self.trainer.max_epochs)
  
#   # Val
#   def validation_step(self, batch, batch_idx) -> Tensor:
#     x, (y_label, y_distance) = batch
#     pred = self.forward(x, 0.0)
#     return self._compute_loss(pred, [x, y_label, y_distance])
  
#   # Test
#   def on_test_start(self):
#     self.test_errors.clear()
#     self.test_labels.clear()
    
#   def test_step(self, batch, batch_idx):
#     x, (y_anomaly, y_distance) = batch
    
#     reconstructed, distance = self.forward(x, 0.0)
#     for input, output, label in zip(x, reconstructed, y_anomaly):
#       self.test_errors.append(
#         self._compute_reconstruction_error(input, output).detach().cpu().numpy()
#       )
#       self.test_labels.append(
#         label.detach().cpu().numpy()
#       )
    
#     return reconstructed.detach().cpu().numpy()
    
#   def on_test_end(self):
#     precision, recall, thresholds = precision_recall_curve(self.test_labels, self.test_errors)
#     f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
#     best_idx = np.argmax(f1_scores)
#     self.save_hyperparameters(threshold=thresholds[best_idx]) # Save best anomaly threshold as a hyperparameter

#   # Predict
#   def predict_step(self, batch, batch_idx):
#     x, (y_label, y_distance) = batch
#     reconstructed, distances = self.forward(x, 0.0)
#     labels = F.mse_loss(x, reconstructed) > self.hparams['threshold']
#     return labels, distances
      

#   def _compute_loss(self, pred, target):
#     reconstructed, pred_dist = pred
#     x, true_label, true_dist = target
    
#     pred_dist = torch.masked_select(pred_dist, true_label == 1)
#     true_dist = torch.masked_select(true_dist, true_label == 1)
    
#     loss_distance = F.mse_loss(pred_dist, true_dist) # calculate distance loss only for true anomalies
#     loss_anomaly = F.mse_loss(reconstructed, x)
    
#     return loss_anomaly #+ self.lambda_distance * torch.nan_to_num(loss_distance)
    
#   def _compute_reconstruction_error(self, x, reconstructed):
#     return F.mse_loss(x, reconstructed, reduction='mean')
  
#   def configure_optimizers(self):
#     self.optimizer = optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
#     self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

#   def forward(self, x, epsilon: float=None) -> Tensor:
#     batch_size, seq_len, _ = x.size()
    
#     # Encode sequence
#     _, h = self.encoder(x)
#     h = self.encoder_layer_norm(h)
    
#     distance = self.distance_head(h)
    
#     # Decoder initialization
#     decoder_input = torch.zeros(batch_size, 1, self.input_dim).to(x.device)
#     decoder_hidden = h
#     outputs = []
    
#     for t in range(seq_len):
      
#       output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
      
#       decoder_hidden = self.decoder_layer_norm(decoder_hidden)
      
#       output = self.output_layer(output)
#       outputs.append(output)
      
#       use_teacher_forcing = epsilon if epsilon is not None else torch.rand(1).item() < self.epsilon
#       decoder_input = x[:, t, :].unsqueeze(1) if use_teacher_forcing else output
      
#     return torch.cat(outputs, dim=1), distance