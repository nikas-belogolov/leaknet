import torch
from tqdm.notebook import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from config import *
from models.model import BaseModel
from os import path

def fit_model(model: BaseModel, train_loader, val_loader, epochs=EPOCHS):
    
    model.configure_optimizers()
    
    model.save_hyperparameters(batch_size=BATCH_SIZE)
    
    print(model.hparams)
    
    history = { 'epoch': [], "train_loss": [], "val_loss": [] }
    best_val_loss = 1_000_000
    
    for epoch in range(epochs):
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit=" batches") as pbar:

            # Metrics
            metrics = {
                'loss': 0.0,
                'val_loss': 0.0
            }
            
            # Training
            model.train()
            for batch in train_loader:
                model.optimizer.zero_grad()
                loss = model.training_step(batch)
                loss.backward()
                model.optimizer.step()
                metrics['loss'] += loss.item()
                pbar.update()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    loss = model.validation_step(batch)
                    metrics['val_loss'] += loss.item()
            
            model.lr_scheduler.step(metrics['val_loss'])
            
            # Logging
            metrics['loss'] /= len(train_loader)
            metrics['val_loss'] /= len(val_loader)
            pbar.set_postfix(metrics)
            
            # History
            history['epoch'].append(epoch)
            history['train_loss'].append(metrics['loss'])
            history['val_loss'].append(metrics['val_loss'])
            
            # Checkpoint
            if metrics['val_loss'] < best_val_loss:
                best_val_loss = metrics['val_loss']
            
                checkpoint_data = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": model.optimizer.state_dict(),
                    "epoch": epoch,
                    "history": history,
                    "hyperparameters": model.hparams
                }
                
                checkpoints_path = path.join(CHECKPOINTS_DIR,
                                            f"{type(model).__name__}.pth")
                
                torch.save(checkpoint_data, checkpoints_path)
            

def save_model(model, history):
    model_path = path.join(MODELS_DIR, f"{type(model).__name__}.pth")
    
    

            
# def cross_val(dataset, n_splits):
#     kf = KFold(n_splits)
#     for i, (train_index, val_index) in enumerate(kf.split(dataset)):
        
#         train_set = Subset(dataset, train_index)
#         val_set = Subset(dataset, val_index)
        
#         train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
#         val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
        
        # Epoch iterations here