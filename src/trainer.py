import torch
from tqdm.notebook import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from config import *
from module import BaseModule
from os import path
from abc import ABC
import numpy as np

import optuna
from optuna.trial import Trial

class SaveBestTrialCallback:
    def __init__(self, save_dir):
        self.save_dir = save_dir
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        # Check if this trial has the best result so far
        if study.best_trial.number == trial.number:
            
            model_path = path.join(self.save_dir, trial.user_attrs["model_name"] + ".pt")
            
            torch.save({
                "model_state_dict": trial.user_attrs["model_state_dict"],
                "optimizer_state_dict": trial.user_attrs["optimizer_state_dict"],
                "history": trial.user_attrs["history"],
                "hyperparameters": trial.user_attrs["hyperparameters"],
            }, model_path)


class Callback(ABC):
    """Base class for all callbacks."""
    def on_validation_end(self, trainer: "Trainer"):
        pass
    
    def on_validation_epoch_end(self, trainer: "Trainer"):
        pass
    
    
class OptunaCallback(Callback):
    """Callback for optuna pruning"""
    def __init__(self, trial: Trial):
        super().__init__()
        
        self.trial = trial
        
    def on_validation_epoch_end(self, trainer):
        self.trial.report(trainer.metrics["val_loss"], step=trainer.current_epoch)
        
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

class Trainer:
    def __init__(self, max_epochs, show_progress_bar, progress_bar_refresh_rate=1):
        self.max_epochs = max_epochs
        self.show_progress_bar = show_progress_bar
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        self.history = { 'epoch': [], "train_loss": [], "val_loss": [] }
        
        self.current_epoch = 0
        self.metrics = {
            'loss': 0.0,
            'val_loss': 0.0
        }

    def test(self, model: BaseModule, dataloader: DataLoader):
        
        model.eval()
        model.on_test_start()

        outputs = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                output = model.test_step(batch, batch_idx)
                output = self._handle_outputs(output)
                outputs.append(output)
                
        model.on_test_end()
                
        return self._aggregate_outputs(outputs)
    
    def predict(self, model: BaseModule, batch):
        
        model.eval()
        outputs = []
        
        with torch.no_grad():
            output = model.predict_step(batch)
            output = self._handle_outputs(output)
            outputs.append(output)

        return self._aggregate_outputs(outputs)
    
    def _handle_outputs(self, output):
        """Convert tensors to CPU/numpy for easier handling"""
        if isinstance(output, torch.Tensor):
            return output.cpu().numpy()
        elif isinstance(output, (list, tuple)):
            return type(output)(self._handle_outputs(x) for x in output)
        elif isinstance(output, dict):
            return {k: self._handle_outputs(v) for k, v in output.items()}
        return output
    
    def _aggregate_outputs(self, outputs):
        """Combine batch outputs into single structure"""
        if isinstance(outputs[0], torch.Tensor):
            return torch.cat(outputs, dim=0)
        elif isinstance(outputs[0], np.ndarray):
            return np.concatenate(outputs, axis=0)
        elif isinstance(outputs[0], (list, tuple)):
            return type(outputs[0])(x for batch in outputs for x in batch)
        elif isinstance(outputs[0], dict):
            keys = outputs[0].keys()
            return {k: torch.cat([out[k] for out in outputs], dim=0) for k in keys}
        return outputs
    
    def _call_callbacks(self, callbacks, hook_name, *args, **kwargs):
        for callback in callbacks:
            method = getattr(callback, hook_name, None)
            if method:
                method(self, *args, **kwargs)

    def fit(self, model: BaseModule, train_dataloaders: DataLoader, val_dataloaders: DataLoader = None, callbacks: list[Callback] = []):
        
        self.train_loader = train_dataloaders
        
        model.trainer = self
        
        optimizers = model.configure_optimizers()         
        optimizer = optimizers['optimizer']
        lr_scheduler = optimizers['lr_scheduler']
        
        model.on_fit_start()
        
        for epoch_idx in range(self.max_epochs):
            
            self.current_epoch = epoch_idx + 1
            
            if self.show_progress_bar and self.current_epoch % self.progress_bar_refresh_rate == 0:
                pbar = tqdm(total=len(train_dataloaders), unit="batches", desc=f"Epoch {epoch_idx + 1}/{self.max_epochs}", leave=True)
            else:
                pbar = None
            
            # Metrics
            self.metrics['loss'] = 0.0
            self.metrics['val_loss'] = 0.0
        
            # Training
            model.train()
            for batch in train_dataloaders:
                
                optimizer.zero_grad()
                
                loss = model.training_step(batch, epoch_idx)
                loss.backward()
                
                optimizer.step()
                
                if lr_scheduler['interval'] == 'step':
                    lr_scheduler['scheduler'].step()
                
                self.metrics['loss'] += loss.item()
                if pbar is not None: pbar.update()
                
            model.on_train_epoch_end()
            
            # Evaluation
            if val_dataloaders:
                model.eval()
                with torch.no_grad():
                    for batch in val_dataloaders:
                        loss = model.validation_step(batch, epoch_idx)
                        self.metrics['val_loss'] += loss.item()
            
            
            # Logging
            self.metrics['loss'] /= len(train_dataloaders)
            self.metrics['val_loss'] /= len(val_dataloaders)
            
            if lr_scheduler['interval'] == 'epoch':
                lr_scheduler['scheduler'].step(self.metrics['val_loss'])
            
            if pbar is not None: pbar.set_postfix(self.metrics)
            
            # History
            self.history['epoch'].append(epoch_idx)
            self.history['train_loss'].append(self.metrics['loss'])
            self.history['val_loss'].append(self.metrics['val_loss'])
            
            self._call_callbacks(callbacks, "on_validation_epoch_end")
            
            if pbar is not None: pbar.close()
            
            model.on_epoch_end()