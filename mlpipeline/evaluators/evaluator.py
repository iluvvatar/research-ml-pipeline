from typing import Dict, Any
from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        raise
