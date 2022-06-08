import torch
from torch import nn
from typing import Union, List
import numpy as np
from collections import OrderedDict

from .loss import Loss


class CrossEntropyLoss(Loss):
    def __init__(self, *,
                 weights=None, label_smoothing=0, reduction='mean'):
        """
        Parameters
        ----------
        weights : Union[List[float], np.ndarray]
        reduction : str
        label_smoothing : float
        """
        assert reduction in ['none', 'mean', 'sum']
        super().__init__()
        self.reduction = reduction
        if weights is None:
            self.weights = None
        else:
            self.weights = nn.Parameter(
                torch.tensor(weights, dtype=torch.float),
                requires_grad=False
            )
        self.label_smoothing = label_smoothing

        if self.weights is None:
            self.cross_entropy = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                reduction=reduction
            )
        else:
            self.cross_entropy = nn.CrossEntropyLoss(
                weight=self.weights.data,
                label_smoothing=label_smoothing,
                reduction=reduction
            )

    def forward(self, *, input: torch.Tensor, target: torch.Tensor):
        return self.cross_entropy(input=input, target=target)

    @classmethod
    def load_state_dict(cls, state_dict: OrderedDict, **kwargs):
        return cls(weights=state_dict['weights'],
                   label_smoothing=state_dict['weights'],
                   reduction=state_dict['weights'])

    def state_dict(self, **kwargs) -> OrderedDict:
        state_dict = super().state_dict()
        weights = None if self.weights is None else self.weights.data.tolist()
        state_dict['weights'] = weights
        state_dict['label_smoothing'] = self.label_smoothing
        state_dict['reduction'] = self.reduction
        return state_dict

    def __repr__(self):
        return repr(self.cross_entropy)
