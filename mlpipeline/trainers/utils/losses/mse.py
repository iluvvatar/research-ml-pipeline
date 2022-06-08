import torch
from torch import nn
from collections import OrderedDict

from .loss import Loss


class MSELoss(Loss):
    def __init__(self, *, reduction='mean'):
        """
        Parameters
        ----------
        reduction : str
        """
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, *, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(input=input, target=target)

    @classmethod
    def load_state_dict(cls, state_dict: OrderedDict, **kwargs) -> 'MSELoss':
        return cls(reduction=state_dict['reduction'])

    def state_dict(self) -> OrderedDict:
        state_dict = super().state_dict()
        state_dict['reduction'] = self.reduction
        return state_dict
