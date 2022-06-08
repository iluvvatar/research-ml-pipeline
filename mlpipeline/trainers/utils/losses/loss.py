import torch
from torch.nn import functional as F
from torch import nn
from typing import Iterable, Union, List, Tuple
import numpy as np
from pathlib import Path
import json
import importlib
from collections import OrderedDict
from abc import ABC, abstractmethod
from multimethod import overload

from ....utils import PathLike


LOSSES_MAPPING = {
    'cross_entropy': ['CrossEntropyLoss'],
    'dice': ['DiceLoss'],
    'mse': ['MSELoss'],
    'siam': ['TripleSiameeseCosineDistanceLoss'],
}


def _dispatch_class(class_name: str):
    # Derived from transformers.AutoTokenizer
    for module_name, classes_names in LOSSES_MAPPING.items():
        if class_name in classes_names:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
    return None


def _add_indent(s, num_spaces):
    # Derived from torch.nn.Module
    s = s.split('\n')
    if len(s) == 1:
        return s[0]
    first = s.pop(0)
    s = [(num_spaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class Loss(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__

    def save(self, path: PathLike):
        path = Path(path)
        assert path.suffix == '.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.state_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: PathLike) -> 'Loss':
        with open(path, encoding='utf-8') as f:
            state_dict = json.load(f)
            class_ = _dispatch_class(state_dict['name'])
            if class_ is None:
                raise ValueError(f'Loss class "{state_dict["name"]}" not found.')
            return class_.load_state_dict(state_dict)

    @classmethod
    @abstractmethod
    def load_state_dict(cls, state_dict: OrderedDict, **kwargs):
        raise NotImplementedError

    def state_dict(self, **kwargs) -> OrderedDict:
        return OrderedDict([('name', self.name)])


class LossList(Loss):
    def __init__(self, losses, *, ratios=None, reduction='mean'):
        """
        Parameters
        ----------
        losses : List[Loss]
        ratios : List[float]
        reduction : str
        """
        assert reduction in ['sum', 'mean', 'none']
        assert ratios is None or len(ratios) == len(losses)
        super().__init__()
        self.losses = nn.ModuleList(losses)
        if ratios is None:
            self.ratios = None
        else:
            self.ratios = nn.Parameter(torch.tensor(ratios,
                                                    dtype=torch.float,
                                                    requires_grad=False))
        self.reduction = reduction

    @overload
    def forward(self, **kwargs) -> torch.Tensor:
        losses = []
        for loss_fn in self.losses:
            losses.append(loss_fn(**kwargs))
        losses = torch.stack(losses)
        if self.ratios is not None:
            losses = losses * self.ratios.data
        if self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'mean':
            return losses.mean()
        else:
            return losses

    @overload
    def forward(self, *, args: list = None, kwargs: list = None) -> torch.Tensor:
        """
        Parameters
        ----------
        args : List[List[Any]]
        kwargs : List[Dict[str, Any]]

        Returns
        -------
        float
        """
        if args is None:
            args = [[] for _ in range(len(self.losses))]
        if kwargs is None:
            kwargs = [{} for _ in range(len(self.losses))]
        losses = []
        for args_i, kwargs_i, loss_fn in zip(args, kwargs, self.losses):
            losses.append(loss_fn(*args_i, **kwargs_i))
        losses = torch.stack(losses)
        if self.ratios is not None:
            losses = losses * self.ratios.data
        if self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'mean':
            return losses.mean()
        else:
            return losses

    @classmethod
    def load_state_dict(cls, state_dict: OrderedDict, **kwargs) -> 'LossList':
        classes = [_dispatch_class(loss_state_dict['name']) for loss_state_dict in state_dict['losses']]
        losses = [class_.load_state_dict(sd) for class_, sd in zip(classes, state_dict['losses'])]
        return cls(losses,
                   ratios=state_dict['ratios'],
                   reduction=state_dict['reduction'])

    def state_dict(self, **kwargs) -> OrderedDict:
        state_dict = super().state_dict()
        state_dict['losses'] = [loss.state_dict() for loss in self.losses]
        if self.ratios is None:
            state_dict['ratios'] = None
        else:
            state_dict['ratios'] = self.ratios.data.tolist()
        state_dict['reduction'] = self.reduction
        return state_dict

    def extra_repr(self) -> str:
        ratios = None \
            if self.alpha is not None or self.ratios is None \
            else self.ratios.data.tolist()
        return f'alpha={self.alpha} ratios={ratios} reduction="{self.reduction}"'

    def __repr__(self):
        lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            lines += extra_repr.split('\n')
        for i, module in enumerate(self.losses):
            mod_str = repr(module)
            mod_str = _add_indent(mod_str, 2)
            lines.append(f'({i}): {mod_str}')

        result = self.name + '('
        if lines:
            result += '\n  ' + '\n  '.join(lines) + '\n'
        result += ')'
        return result

    def __getitem__(self, item):
        return self.losses[item]

    def __len__(self):
        return len(self.losses)
