import torch
from abc import ABC, abstractmethod

import torch.optim.optimizer

class BaseModel(torch.nn.Module, ABC):
    
    def __init__(self):
        super().__init__()
        self._criterion = None
        self._optimizer = None
        self._scheduler = None
        self._hparams = {}
        
    @abstractmethod
    def training_step(self, batch) -> torch.Tensor:
        pass
    
    @abstractmethod
    def validation_step(self, batch) -> torch.Tensor:
        pass
    
    # @abstractmethod
    def test_step(self, batch) -> torch.Tensor:
        pass
    
    # @abstractmethod
    def predict_step(self, batch) -> torch.Tensor:
        pass
    
    @abstractmethod
    def configure_optimizers(self):
        pass
        
    @abstractmethod
    def forward(self, x):
        pass
    
    def save_hyperparameters(self, **hparams):
        self._hparams |= hparams # Extend hparams
    
    # PROPERTIES
    
    # Hyperparameters
    @property
    def hparams(self):
        return self._hparams
    
    # Criterion
    @property
    def criterion(self):
        return self._criterion
    
    @criterion.setter
    def criterion(self, criterion):
        self._criterion = criterion

    # Optimizer
    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer: torch.optim.Optimizer):
        self._optimizer = optimizer
        # self._hparams["learning_rate"] = optimizer.param_groups[0]['lr']

    # Scheduler
    @property
    def lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self._lr_scheduler
    
    @lr_scheduler.setter
    def lr_scheduler(self, lr_scheduler):
        self._lr_scheduler = lr_scheduler
