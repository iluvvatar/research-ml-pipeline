import torch
from torch import nn
from abc import abstractmethod, ABC
from pathlib import Path


class Model(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.name: str = self.__class__.__name__

    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError

    def save(self, path: Path):
        torch.save(self.state_dict(), path)

    def load(self, path: Path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    @property
    def device(self):
        param = next(iter(self.parameters()))
        return param.data.device
