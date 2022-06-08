from torch.utils.data import Sampler
from typing import Callable, List
import datasets
import numpy as np

from ..dataset import HFBasedDataset


class UnbalancedEntitiesSampler(Sampler):
    def __init__(self, dataset: HFBasedDataset, *,
                 entities_deserialize_fn: Callable,
                 entities_column: str,
                 tokens_spans_column: str,
                 entity_types_shares: str = 'uniform',
                 size: int = None):
        assert isinstance(dataset.data, datasets.Dataset)
        assert entity_types_shares in ['uniform', 'log']
        super().__init__(dataset)
        self.dataset = dataset
        self.size = len(dataset) if size is None else size
        self.entities_column = entities_column
        self.entities_deserialize_fn = entities_deserialize_fn
        self.n = {ent_type: np.zeros(len(dataset))
                  for ent_type in dataset.entity_types}
        for i, example in enumerate(dataset):
            entities = [entities_deserialize_fn(e) for e in example[entities_column]]
            # Filter entities because of MaxLenSplitCollator doesn't filter
            # entities for each split
            sent_start = example[tokens_spans_column][0][0]
            sent_stop = example[tokens_spans_column][-1][1]
            entities = [e for e in entities if sent_start < e.stop and e.start < sent_stop]
            for e in entities:
                self.n[e.type][i] += 1
        self.N = {ent_type: n_i.sum() for ent_type, n_i in self.n.items()}
        if sum(self.N.values()) == 0:
            raise ValueError(f'No one entities provided in dataset.')
        if entity_types_shares == 'uniform':
            self.shares = {ent_type: 1
                           for ent_type, N_i in self.N.items()}
        elif entity_types_shares == 'log':
            self.shares = {ent_type: np.log(1 + N_i)
                           for ent_type, N_i in self.N.items()}
        self.probs = {ent_type: self.n[ent_type] / self.N[ent_type]
                      if self.N[ent_type] != 0 else np.zeros(len(dataset))
                      for ent_type in self.N}
        self.entity_types_sampled_count = {ent_type: 0
                                           for ent_type in dataset.entity_types}
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        for _ in range(len(self)):
            ent_type = self.undersampled_ent_type()
            index = np.random.choice(self.indices, p=self.probs[ent_type])
            self.entity_types_sampled_count[ent_type] += self.n[ent_type][index]
            yield int(index)

    def __len__(self):
        return self.size

    def undersampled_ent_type(self, except_types: List[str] = ()):
        def key_fn(ent_type):
            if ent_type in except_types or self.shares[ent_type] == 0:
                return np.inf
            else:
                return self.entity_types_sampled_count[ent_type] / self.shares[ent_type]
        return min(self.entity_types_sampled_count, key=key_fn)


class UnbalancedEntitiesSamplerForTripleSiameese(UnbalancedEntitiesSampler):
    def __iter__(self):
        """
        Returns
        -------
        int
            Index of "anchor" sample
        int
            Index of sample containing the same entity type as "anchor"
        int
            Index of sample containing contrastive entity type
        str
            "Anchor" entity type
        str
            Contrastive entity type
        """
        for _ in range(len(self)):
            ent_type = self.undersampled_ent_type()
            contr_ent_type = self.undersampled_ent_type(except_types=[ent_type])
            index1 = int(np.random.choice(self.indices,
                                          p=self.probs[ent_type]))
            index2 = index1
            i = 0
            while index2 == index1:
                index2 = int(np.random.choice(self.indices,
                                              p=self.probs[ent_type]))
                i += 1
                if i == 1_000:
                    raise ValueError('Couldn\'t select sample with same type as anchor sample')
            index3 = index2
            i = 0
            while index3 == index1 or index3 == index2:
                index3 = int(np.random.choice(self.indices,
                                              p=self.probs[contr_ent_type]))
                i += 1
                if i == 1_000:
                    raise ValueError('Couldn\'t select sample with different type from anchor sample')
            self.entity_types_sampled_count[ent_type] += self.n[ent_type][index1]
            self.entity_types_sampled_count[ent_type] += self.n[ent_type][index2]
            self.entity_types_sampled_count[ent_type] += self.n[ent_type][index3]

            yield index1, index2, index3, ent_type, contr_ent_type
