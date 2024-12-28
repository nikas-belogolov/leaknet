import torch
from abc import ABC, abstractmethod, abstractproperty



class Model(torch.nn.Module, ABC):
    
    def __init__(self):
        super(Model, self).__init__()
        self._criterion = None
        self._optimizer = None
        self._scheduler = None
        
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
    def scheduler(self):
        return self._scheduler
    
    @scheduler.setter
    def scheduler(self, scheduler):
        self._scheduler = scheduler
