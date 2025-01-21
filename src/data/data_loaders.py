from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from config import TRAIN_SIZE, VAL_SIZE, BATCH_SIZE

def _get_data_loaders(train_set, val_set, test_set, batch_size):
  train_loader = DataLoader(train_set, batch_size, shuffle=True)
  val_loader = DataLoader(val_set, batch_size, shuffle=True)
  test_loader = DataLoader(test_set, batch_size, shuffle=False)
  
  return train_loader, val_loader, test_loader
      
def get_classifier_data_loaders(dataset, batch_size):
  train_size = int(TRAIN_SIZE * len(dataset))
  val_size = int(VAL_SIZE * len(dataset))
  test_size = len(dataset) - train_size - val_size

  train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
  
  return _get_data_loaders(train_set, val_set, test_set, batch_size)
        
def get_autoencoder_data_loaders(dataset, batch_size):
  normal_dataset = Subset(dataset, dataset.normal_indices)
  anomalous_dataset = Subset(dataset, dataset.anomalous_indices)

  train_size = int(TRAIN_SIZE * len(normal_dataset))
  val_size = int(VAL_SIZE * len(normal_dataset))
  test_size = len(normal_dataset) - train_size - val_size

  train_set, val_set, test_set = random_split(normal_dataset, [train_size, val_size, test_size]) # Split train/val sets which contain only normal data
  test_set = ConcatDataset([anomalous_dataset, test_set]) # Create set for testing, consisting of all anomalous dataset and some samples from normal dataset
  
  return _get_data_loaders(train_set, val_set, test_set, batch_size)


