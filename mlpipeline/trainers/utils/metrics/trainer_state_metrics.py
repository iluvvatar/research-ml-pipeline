from abc import ABC, abstractmethod
from typing import List, Any, Dict, Iterable, Callable
from collections import OrderedDict

from .metric import Metric
from ..losses import LossList


# class LossMetric(Metric):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.loss = 0
#
#     def add(self, example: Dict[str, Any]):
#         raise NotImplementedError
#
#     def add_batch(self, batch: Dict[str, List[Any]]):
#         self.loss += float(sum(batch[self.name]) / len(batch[self.name]))
#
#     def compute(self, trainer):
#         return {self.name: self.loss}
#
#     def clear(self):
#         self.loss = 0
#
#     @property
#     def names(self):
#         return [self.name]


class LossListAlphaMetric(Metric):
    def __init__(self, name: str = 'loss_list_alpha'):
        super().__init__(name=name)

    def add(self, example: Dict[str, Any]):
        pass

    def add_batch(self, batch: Dict[str, List[Any]]):
        pass

    def compute(self, trainer):
        assert isinstance(trainer.loss_fn, LossList)
        return OrderedDict({self.name: trainer.loss_fn.ratios[0].item()})

    def clear(self):
        pass


class LRMetric(Metric):
    def __init__(self, name: str = 'lr', names: List[str] = None):
        super().__init__(name=name)
        if names is None:
            self._names = [name]
        else:
            self._names = names

    def add(self, example: Dict[str, Any]):
        pass

    def add_batch(self, batch: Dict[str, List[Any]]):
        pass

    def compute(self, trainer):
        params = trainer.optimizer.param_groups
        assert len(self.names) == len(params)
        return OrderedDict([(name, param['lr'])
                            for name, param in zip(self.names, params)])

    def clear(self):
        pass

    @property
    def names(self):
        return self._names
