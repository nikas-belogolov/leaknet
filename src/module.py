import torch

from abc import ABC, abstractmethod

from torch import nn, ScriptModule
import torch.optim.optimizer
import inspect

from typing import Any, Union

class BaseModule(nn.Module, ABC):
    
    hparams = {}
    optimizers = None
    lr_schedulers = None
    
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        pass
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self.training_step(batch, batch_idx)
    
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        pass
    
    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        batch = kwargs.get("batch", args[0])
        return self(batch)
    
    @abstractmethod
    def configure_optimizers(self):
        pass
    
    @torch.no_grad()
    def to_torchscript(self, **kwargs: Any) -> Union[ScriptModule, dict[str, ScriptModule]]:
        torchscript_module = torch.jit.script(self.eval(), **kwargs)
        return torchscript_module
    
    def save_hyperparameters(self, **hparams):

        if hparams:
            self.hparams.update(hparams)
            return

        frame = inspect.currentframe().f_back  # Get the previous frame (where __init__ is called)
        args, _, _, local_vars = inspect.getargvalues(frame)

        self.hparams = {arg: local_vars[arg] for arg in args if arg != "self"}

        if 'kwargs' in local_vars and local_vars['kwargs']:
            self.hparams.update(**local_vars['kwargs'])  # Flatten **kwargs into hparams
            
        
                    
    # EVENTS
    def on_fit_start(self):
        pass
    
    def on_epoch_start(self):
        pass
    
    def on_epoch_end(self):
        pass
    
    def on_train_epoch_start(self):
        pass
    
    def on_train_epoch_end(self):
        pass
    
    def on_test_start(self):
        pass
    
    def on_test_end(self):
        pass
    
    # Optimizer
    # @torch.jit.unused
    # @property
    # def optimizer(self) -> torch.optim.Optimizer:
    #     return self._optimizer

    # @torch.jit.unused
    # @optimizer.setter
    # def optimizer(self, optimizer: torch.optim.Optimizer):
    #     self._optimizer = optimizer

    # # Scheduler
    # @torch.jit.unused
    # @property
    # def lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
    #     return self._lr_scheduler
    
    # @torch.jit.unused
    # @lr_scheduler.setter
    # def lr_scheduler(self, lr_scheduler):
    #     self._lr_scheduler = lr_scheduler
