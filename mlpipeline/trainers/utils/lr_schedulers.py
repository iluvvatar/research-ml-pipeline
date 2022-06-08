import torch
from abc import ABC, abstractmethod
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LRScheduler(ABC):
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    @abstractmethod
    def step(self, trainer):
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict):
        raise NotImplementedError


class NoneScheduler(LRScheduler):
    def step(self, trainer):
        pass

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        return self


class PlateauScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, *args, **kwargs):
        super().__init__(optimizer)
        self.scheduler = ReduceLROnPlateau(optimizer, *args, **kwargs)

    def step(self, trainer):
        metric_name = trainer.key_metric_name
        metric_val = trainer.history.metrics[metric_name][-1]
        self.scheduler.step(metric_val)

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)
