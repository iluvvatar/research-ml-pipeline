from typing import List, Union, Callable
from torch.utils.data import IterableDataset

import numpy as np

from ..dataset import HFBasedDataset
from .samplers import UnbalancedEntitiesSamplerForTripleSiameese
from ...utils import PathLike
from ...processors.nlp.labelizers import BILOULabelizer


class TripleSimaeeseDatasetWrapper(IterableDataset):
    def __init__(self, dataset: HFBasedDataset, *,
                 labelizer: BILOULabelizer,
                 entity_types_shares: str = 'uniform',
                 out_entity_mask_column: str = 'entity_mask',
                 out_entity_type_id_column: str = 'siam_entity_type_id',
                 size: int = None):
        self.dataset = dataset
        self.labelizer = labelizer
        self.size = size
        self.sampler = UnbalancedEntitiesSamplerForTripleSiameese(
            dataset,
            entities_deserialize_fn=labelizer.entities_deserialize_fn,
            entities_column=labelizer.entities_column,
            tokens_spans_column=labelizer.tokens_spans_column,
            entity_types_shares=entity_types_shares,
            size=size
        )
        self.out_entity_mask_column = out_entity_mask_column
        self.out_entity_type_id_column = out_entity_type_id_column

    def __iter__(self):
        for index1, index2, index3, ent_type, contr_ent_type in self.sampler:
            sample1 = self.dataset[index1]
            sample2 = self.dataset[index2]
            sample3 = self.dataset[index3]
            ent_type_id = self.labelizer.ent_type2int[ent_type]
            contr_ent_type_id = self.labelizer.ent_type2int[contr_ent_type]

            labels_ids_column = self.labelizer.out_labels_ids_column
            labels_ids1 = np.array(sample1[labels_ids_column], dtype=int)
            labels_ids2 = np.array(sample2[labels_ids_column], dtype=int)
            labels_ids3 = np.array(sample3[labels_ids_column], dtype=int)
            labels_ids1 = labels_ids1[:, ent_type_id]
            labels_ids2 = labels_ids2[:, ent_type_id]
            labels_ids3 = labels_ids3[:, contr_ent_type_id]

            O_tag_id = self.labelizer.bilou2int['O']
            col = self.out_entity_mask_column
            sample1[col] = (labels_ids1 != O_tag_id).astype(int)
            sample2[col] = (labels_ids2 != O_tag_id).astype(int)
            sample3[col] = (labels_ids3 != O_tag_id).astype(int)

            col = self.out_entity_type_id_column
            sample1[col] = ent_type_id
            sample2[col] = ent_type_id
            sample3[col] = contr_ent_type_id

            yield sample1, sample2, sample3

    def __len__(self):
        return self.size
