import torch
from torch import nn
from typing import List
from collections import OrderedDict

from .loss import Loss


class TripleSiameeseCosineDistanceLoss(Loss):
    def __init__(self, *, margin=1, reduction='mean'):
        """
        Parameters
        ----------
        margin : float
        reduction : str
        """
        assert reduction in ['mean', 'sum', 'none']
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.cos = nn.CosineSimilarity(dim=1)
        self.pooling = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, *,
                hidden_states: List[torch.Tensor],
                masks: List[torch.Tensor]):
        assert 3 == len(hidden_states) == len(masks)
        pooled_states = []
        for i in range(3):
            masked_hs = (hidden_states[i] * masks[i].unsqueeze(-1)).permute(0, 2, 1)
            pooled_states.append(self.pooling(masked_hs).squeeze())
        pos_cos = self.cos(pooled_states[0], pooled_states[1])
        neg_cos = self.cos(pooled_states[0], pooled_states[2])

        zero = torch.zeros(pos_cos.shape, device=pos_cos.device)
        losses = torch.max(neg_cos - pos_cos + self.margin, zero)
        if self.reduction == 'mean':
            return losses.mean()
        if self.reduction == 'sum':
            return losses.sum()
        return losses

    @classmethod
    def load_state_dict(cls, state_dict: OrderedDict, **kwargs):
        return cls(margin=state_dict['margin'],
                   reduction=state_dict['reduction'])

    def state_dict(self, **kwargs) -> OrderedDict:
        state_dict = super().state_dict()
        state_dict['margin'] = self.margin
        state_dict['reduction'] = self.margin
        return state_dict
