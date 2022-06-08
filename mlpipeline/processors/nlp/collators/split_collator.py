from datasets.arrow_dataset import Batch, Example
from multimethod import overload
from typing import List, Dict, Any, Union
import torch
import math
from collections import defaultdict
from abc import abstractmethod

from .collator import Collator
from ....datasets import HFBasedDataset


class SplitCollator(Collator):
    """
        Splits sequences into several sentences and then
        can unite them back after getting model's output.
        """

    def __init__(self, *,
                 collate_columns: List[str],
                 pk_columns: List[str],
                 unite_columns: List[str],
                 out_is_united_column: str = None):
        """
        Parameters
        ----------
        collate_columns : list[str]
            Columns that should be split and than can be united.
        pk_columns : list[str]
            Primary key columns. Parameter name is taken from SQL.
            Columns by that each sample can be identified.
        unite_columns : list[str]
            Columns that will be added to dataset after splitting
            (e.g. model's outputs) and should also be united in postprocessing
            along with split_columns.
        out_is_united_column : str, optional
            Column name in result dataset after postprocessing that will
            contain flag if sentence was split and then united back.
            If not specified tokenizer will not create a column with this flag.

        Split only collate_columns. All other columns' values will be
        copied as is over all split parts. So each split will contain
        identical value in save_columns.
        """
        assert len(collate_columns) > 0
        assert len(set(collate_columns) & set(unite_columns)) == 0
        super().__init__(collate_columns=collate_columns)

        self.pk_columns = pk_columns
        self.unite_columns = unite_columns + self.collate_columns

        # Used in self.unite()
        self.unification_buffer = defaultdict(list)

        self.split_id_col_name = f'<{self.__class__.__name__}>_split_id'
        self.n_splits_col_name = f'<{self.__class__.__name__}>_n_splits'
        self.out_is_united_column = out_is_united_column

    def preprocess(self,
                   dataset: HFBasedDataset,
                   use_cached=True,
                   *args, **kwargs) -> HFBasedDataset:
        return dataset.map(self.collate, batched=True,
                           load_from_cache_file=use_cached,
                           *args, **kwargs)

    def postprocess(self,
                    dataset: HFBasedDataset,
                    use_cached=True,
                    *args, **kwargs) -> HFBasedDataset:
        return dataset.map(self.unite, batched=True,
                           load_from_cache_file=use_cached,
                           remove_columns=(self.n_splits_col_name,
                                           self.split_id_col_name),
                           *args, **kwargs)

    @overload
    def collate(self, example: Example) -> Dict[str, Any]:
        raise NotImplementedError

    @overload
    def collate(self, batch: Batch) -> Dict[str, List[Any]]:
        return self.split(batch)

    @overload
    def collate(self, batch: list) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def split(self, batch: Batch) -> Dict[str, List[Any]]:
        raise NotImplementedError

    def unite(self, batch: Union[Batch, Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
        assert self.n_splits_col_name in batch
        assert self.split_id_col_name in batch

        united_batch = defaultdict(list)
        for sample in HFBasedDataset.batch_samples(batch):
            if sample[self.n_splits_col_name] > 1:
                united_sample = self.update_unification_buffer(sample)
                if united_sample is not None:
                    for col, value in united_sample.items():
                        if col not in (self.n_splits_col_name,
                                       self.split_id_col_name):
                            united_batch[col].append(value)
                    if self.out_is_united_column is not None:
                        united_batch[self.out_is_united_column].append(True)
            else:
                for col, value in sample.items():
                    if col not in (self.n_splits_col_name,
                                   self.split_id_col_name):
                        united_batch[col].append(value)
                if self.out_is_united_column is not None:
                    united_batch[self.out_is_united_column].append(False)
        return dict(united_batch)

    def update_unification_buffer(self, sample: Dict[str, Any]) -> Union[None, Dict[str, Any]]:
        key = tuple(sample[col] for col in self.pk_columns)
        buffer = self.unification_buffer[key]
        buffer.append(sample)
        buffer.sort(
            key=lambda s: s[self.split_id_col_name]
        )
        if len(buffer) == sample[self.n_splits_col_name]:
            united_sample = {col: value for col, value in buffer[0].items()
                             if col not in self.unite_columns}
            for col in self.unite_columns:
                united_sample[col] = []
            for samp in buffer:
                for col in self.unite_columns:
                    united_sample[col].extend(samp[col])
            del self.unification_buffer[key]
            return united_sample
        return None


class MaxLenSplitCollator(SplitCollator):
    """
    Splits sequences that exceed max_len into several sentences and then
    can unite them back after getting model's output.
    """
    def __init__(self,
                 max_len: int,
                 *args, **kwargs):
        assert max_len > 0
        super().__init__(*args, **kwargs)
        self.max_len = max_len

    @overload
    def split(self, batch: Batch) -> Dict[str, List[Any]]:
        self.validate_batch(batch)
        collated_batch = defaultdict(list)
        for sample in HFBasedDataset.batch_samples(batch):
            length = len(sample[self.collate_columns[0]])
            n_splits = math.ceil(length / self.max_len)
            for i in range(n_splits):
                start = self.max_len * i
                stop = self.max_len * (i + 1)
                for col in sample.keys():
                    if col in self.collate_columns:
                        collated_batch[col].append(sample[col][start:stop])
                    else:
                        collated_batch[col].append(sample[col])
                collated_batch[self.n_splits_col_name].append(n_splits)
                collated_batch[self.split_id_col_name].append(i)
        return dict(collated_batch)

    def validate_batch(self, batch: Batch):
        lengths = [len(seq) for seq in batch[self.collate_columns[0]]]
        for col in self.collate_columns:
            lengths_i = [len(seq) for seq in batch[col]]
            assert lengths_i == lengths, f'{lengths_i}\n{lengths}\n\n' \
                                         f'{self.collate_columns[0]}\n{batch[self.collate_columns[0]]}\n\n' \
                                         f'{col}\n{batch[col]}'



