import torch
from torch.nn import functional as F
from torch import nn
from typing import Union, List
import numpy as np
from collections import OrderedDict

from .loss import Loss


class DiceLoss(Loss):
    def __init__(self, *,
                 weights=None, adjusted=False, adj_alpha=0.5,
                 label_smoothing=0, reduction='mean'):
        """
        Parameters
        ----------
        weights : List[int] | List[float] | np.ndarray
            Class weights for imbalanced classes
        adjusted : bool
            See https://arxiv.org/pdf/1911.02855.pdf equation 12.
        adj_alpha : float
            Used in self-adjusted dice loss.
            See https://arxiv.org/pdf/1911.02855.pdf equation 12.
        label_smoothing : float
        reduction : str
            Loos reduction method over batch.
        """
        assert reduction in ['none', 'mean', 'sum']
        super().__init__()
        self.eps = 1e-6
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.adjusted = adjusted
        self.adj_alpha = adj_alpha
        if weights is None:
            self.weights = None
        else:
            self.weights = nn.Parameter(
                torch.tensor(weights, dtype=torch.float),
                requires_grad=False
            )

    def forward(self, *,
                input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Derived from https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

        Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.

        Parameters
        ----------
            target : torch.Tensor
                A tensor of shape [B, 1, ...].
            input : torch.Tensor
                A tensor of shape [B, C, ...].
                Raw logits model's output.

        Returns
        -------
            torch.Tensor
        """
        num_classes = input.shape[1]
        ndim = input.ndim
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[target.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, ndim-1, *range(1, ndim-1)).float()
            true_1_hot_f = true_1_hot[:, 0:1]
            true_1_hot_s = true_1_hot[:, 1:2]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(input)
            neg_prob = 1 - pos_prob
            probs = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, ndim-1, *range(1, ndim-1)).float()
            probs = F.softmax(input, dim=1)
        true_1_hot = true_1_hot.type(input.type())
        true_1_hot = true_1_hot * (1 - self.label_smoothing)
        dims = tuple(range(2, probs.ndim))
        if self.adjusted:
            adj = (1 - probs + self.eps) ** self.adj_alpha
        else:
            adj = 1
        intersection = torch.sum(adj * probs * true_1_hot, dims)
        cardinality = torch.sum(adj * probs ** 2 + true_1_hot ** 2, dims)
        dice = (2. * intersection + self.eps) / (cardinality + self.eps)
        if self.weights is not None:
            dice = dice * self.weights.data
        dice = dice.mean(dim=1)
        if self.reduction == 'none':
            return 1 - dice
        if self.reduction == 'mean':
            return (1 - dice).mean()
        if self.reduction == 'sum':
            return (1 - dice).sum()

    @classmethod
    def load_state_dict(cls, state_dict: OrderedDict, **kwargs):
        return cls(weights=state_dict['weights'],
                   adjusted=state_dict['adjusted'],
                   adj_alpha=state_dict['adj_alpha'],
                   label_smoothing=state_dict['label_smoothing'],
                   reduction=state_dict['reduction'])

    def state_dict(self, **kwargs) -> OrderedDict:
        state_dict = super().state_dict()
        weights = None if self.weights is None else self.weights.data.tolist()
        state_dict['weights'] = weights
        state_dict['adjusted'] = self.adjusted
        state_dict['adj_alpha'] = self.adj_alpha
        state_dict['label_smoothing'] = self.label_smoothing
        state_dict['reduction'] = self.reduction
        return state_dict
