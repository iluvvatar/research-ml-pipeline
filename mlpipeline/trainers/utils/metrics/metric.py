from abc import ABC, abstractmethod
from typing import List, Any, Dict
from collections import OrderedDict


class Metric(ABC):
    def __init__(self, *, name: str = None):
        self.name = self.__class__.__name__ if name is None else name

    @abstractmethod
    def add(self, example: Dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def add_batch(self, batch: Dict[str, List[Any]]):
        raise NotImplementedError

    @abstractmethod
    def compute(self, trainer) -> OrderedDict:
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        raise NotImplementedError

    @property
    def names(self) -> List[str]:
        return [self.name]
