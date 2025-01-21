from torch.utils.data import Dataset
from os import listdir, path
import pandas as pd
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import re
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class LeakAnomalyDetectionDataset(Dataset):
    def __init__(self, normal_data_dir, anomalous_data_dir, withDistance=True):

      self.data = []
      self.anomalies = []
      self.distances = []

      for files_dir, label in zip([normal_data_dir, anomalous_data_dir], range(2)):
        for file in listdir(files_dir):
          df = pd.read_csv(path.join(files_dir, file))
          df = df.set_index('timestamp')
          df = df.sort_index()
          df["delta"] = np.abs(df["pressure"] - df["flow"])
          df = df[["delta", "pressure", "flow"]]
          
          distance = int(re.search('\d+(?=m)', file).group())
          
          
          # print(label, distance)
          
          # target = torch.tensor()
          
          # print(target)
          
          self.data.append(torch.tensor(df.to_numpy(dtype=np.float32), dtype=torch.float32))
          self.distances.append(distance)
          self.anomalies.append(label)
          # self.labels.append([label, distance])
      
      self._num_features = self.data[0].size(-1)
      
      # remove pad_sequence, pad instead in collate_fn
      self.data = pad_sequence(self.data, batch_first=True, padding_value=0)
      self.distances = torch.tensor(self.distances, dtype=torch.float32)
      self.anomalies = torch.tensor(self.anomalies, dtype=torch.float32)
      # self.labels = torch.tensor(self.labels, dtype=torch.float32)
      
      self.scaler = StandardScaler()
      self.data = self.scaler.fit_transform(
                    self.data.reshape((-1, self.data.size(-1)))
                  ).reshape(len(self.data), -1, self.data.size(-1))
      
      self.data = torch.tensor(self.data, dtype=torch.float32)
      
      # delta feature
      # delta = np.abs(self.data[:, :, 0] - self.data[:, :, 1]).unsqueeze(-1)
      # self.data = torch.cat((self.data, delta), dim=-1)
      
      self.normal_indices = [i for i, label in enumerate(self.anomalies) if label == 0]
      self.anomalous_indices = [i for i, label in enumerate(self.anomalies) if label == 1]

    def __len__(self):
      return len(self.data)

    def __getitem__(self, idx):
      sequence = self.data[idx]
      anomaly_label = self.anomalies[idx]
      distance_label = self.distances[idx]
      # label = self.labels[idx]
      return sequence, (anomaly_label, distance_label) # return here also the length of the sequence
    
    @property
    def num_features(self):
        return self._num_features
      
# def collate_fn(batch):
#     # Extract sequences, labels, and lengths
#     sequences, labels, lengths = zip(*batch)
#     lengths = torch.tensor(lengths)
    
#     # Sort by lengths in descending order
#     lengths, sort_idx = lengths.sort(descending=True)
#     sequences = [sequences[i] for i in sort_idx]
#     labels = torch.tensor([labels[i] for i in sort_idx])
    
#     # Pad sequences
#     padded_sequences = pad_sequence(sequences, batch_first=True)
    
#     return padded_sequences, labels, lengths

# print(summary(model, input_size=train_loader.dataset[0][0].shape))