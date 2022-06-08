from abc import ABC, abstractmethod
from typing import List, Any, Dict, Iterable, Callable
from collections import OrderedDict

from .metric import Metric
from ....datasets.nlp.units import Entity
from ....datasets import HFBasedDataset


class F1MacroScoreNER(Metric):
    def __init__(self, *,
                 entity_types: List[str],
                 entities_column: str,
                 predicted_entities_column: str,
                 entities_deserialize_fn: Callable[[str], Entity] = Entity.from_brat,
                 name='f1-macro',
                 return_detailed: bool = False):
        super().__init__(name=name)
        self.entity_types = entity_types
        self.return_detailed = return_detailed

        self.entities_column = entities_column
        self.predicted_entities_column = predicted_entities_column

        self.entities_deserialize_fn = entities_deserialize_fn

        self.tps = [0] * len(entity_types)
        self.fps = [0] * len(entity_types)
        self.fns = [0] * len(entity_types)

    def add(self, example: Dict[str, Any]):
        entities = [self.entities_deserialize_fn(e) for e in example[self.entities_column]]
        pred_entities = [self.entities_deserialize_fn(e) for e in example[self.predicted_entities_column]]
        entities = [(e.start, e.stop, e.type) for e in entities]
        pred_entities = [(e.start, e.stop, e.type) for e in pred_entities]
        for i, type_ in enumerate(self.entity_types):
            real = set(filter(lambda e: e[2] == type_, entities))
            pred = set(filter(lambda e: e[2] == type_, pred_entities))

            self.tps[i] += len(real & pred)
            self.fps[i] += len(pred - real)
            self.fns[i] += len(real - pred)

    def add_batch(self, batch: Dict[str, List[Any]]):
        for example in HFBasedDataset.batch_samples(batch):
            self.add(example)

    def compute(self, trainer) -> OrderedDict:
        precisions = [tp / (tp + fp) if (tp + fp) != 0 else 0 for tp, fp in zip(self.tps, self.fps)]
        recalls = [tp / (tp + fn) if (tp + fn) != 0 else 0 for tp, fn in zip(self.tps, self.fns)]
        f1 = [2 * p * r / (p + r) if (p + r) != 0 else 0 for p, r in zip(precisions, recalls)]
        result = OrderedDict({self.name: round(sum(f1) / len(f1), 5)})
        if self.return_detailed:
            for i, ent_type in enumerate(self.entity_types):
                result[f'f1-{ent_type}'] = f1[i]
                result[f'prc-{ent_type}'] = precisions[i]
                result[f'rcl-{ent_type}'] = recalls[i]
        return result

    def clear(self):
        self.tps = [0] * len(self.entity_types)
        self.fps = [0] * len(self.entity_types)
        self.fns = [0] * len(self.entity_types)

    @property
    def names(self):
        names = [self.name]
        if self.return_detailed:
            for ent_type in self.entity_types:
                names += [f'f1-{ent_type}', f'prc-{ent_type}', f'rcl-{ent_type}']
        return names
