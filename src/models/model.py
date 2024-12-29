import torch
from abc import ABC, abstractmethod



class Model(torch.nn.Module, ABC):
    
    def __init__(self):
        super(Model, self).__init__()
        self._criterion = None
        self._optimizer = None
        self._scheduler = None
        
    @abstractmethod
    def training_step(self, batch) -> torch.Tensor:
        pass
    
    @abstractmethod
    def validation_step(self, batch) -> torch.Tensor:
        pass
    
    @abstractmethod
    def configure_optimizers(self):
        pass
        
    @abstractmethod
    def forward(self, x):
        pass
    
    # Criterion
    @property
    def criterion(self):
        return self._criterion
    
    @criterion.setter
    def criterion(self, criterion):
        self._criterion = criterion

    # Optimizer
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    # Scheduler
    @property
    def lr_scheduler(self):
        return self._lr_scheduler
    
    @lr_scheduler.setter
    def lr_scheduler(self, lr_scheduler):
        self._lr_scheduler = lr_scheduler
