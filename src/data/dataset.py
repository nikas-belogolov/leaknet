from torch.utils.data import Dataset
from os import listdir, path
import pandas as pd
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from sklearn.preprocessing import MinMaxScaler

class LeakAnomalyDetectionDataset(Dataset):
    def __init__(self, normal_data_dir, anomalous_data_dir):

      self.data = []
      self.labels = []

      for files_dir, label in zip([normal_data_dir, anomalous_data_dir], range(2)):
        for file in listdir(files_dir):
          df = pd.read_csv(path.join(files_dir, file))
          df = df.set_index('timestamp')
          df = df.sort_index()
          # df["delta"] = np.abs(df["pressure"] - df["flow"])
          
          self.data.append(torch.tensor(df.to_numpy(dtype=np.float32), dtype=torch.float32))
          self.labels.append(label)
      
      self.data = pad_sequence(self.data, batch_first=True, padding_value=0)
      self.labels = torch.tensor(self.labels, dtype=torch.float32)
      
      self.scaler = MinMaxScaler()
      self.data = self.scaler.fit_transform(
                    self.data.reshape((-1, self.data.size(-1)))
                  ).reshape(len(self.data), -1, self.data.size(-1))
      print(self.data.shape)
      self.data = torch.tensor(self.data, dtype=torch.float32)
      delta = np.abs(self.data[:, :, 0] - self.data[:, :, 1]).unsqueeze(-1)
      self.data = torch.cat((self.data, delta), dim=-1)
      

    def __len__(self):
      return len(self.data)

    def __getitem__(self, idx):
      segment = self.data[idx]
      label = self.labels[idx]


      return segment, label