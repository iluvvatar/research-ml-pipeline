from typing import List, Dict, Any, Tuple, Union, Set
import torch
import numpy as np
from multimethod import overload
from datasets.arrow_dataset import Example, Batch

from .collator import Collator
from ....datasets import HFBasedDataset


class PaddingCollator(Collator):
    """
    Is used in DataLoader(collate_fn=PaddingCollator().collate).
    """
    def __init__(self, *,
                 collate_columns: List[str],
                 pad_value: int,
                 padding_type: str = 'longest',
                 max_len: int = None):
        """
        Parameters
        ----------
        collate_columns : list[str]
            Columns that should be padded.
        pad_value : int
            Padding fill be filled by pad_value.
        padding_type : list[str], optional
            If 'longest', will pad up to longest sequence in batch.
            If 'max_length', will pad all sequences up to max_len.
        max_len : int, optional
            If padding_type='max_length', will pad all sequences up to max_len.
        """
        assert padding_type in ['longest', 'max_length']
        assert max_len is None or max_len > 0
        assert not (max_len is None and padding_type == 'max_length')
        super().__init__(collate_columns=collate_columns)

        self.pad_value = pad_value
        self.padding_type = padding_type
        self.max_len = max_len

    @overload
    def collate(self, example: Example) -> Dict[str, Any]:
        assert self.padding_type == 'max_length'

        padded_example = {}
        for col in self.collate_columns:
            sequence = np.array(example[col])
            item_shape = sequence.shape[1:]
            pad_seq = np.full((self.max_len - len(sequence), *item_shape),
                              self.pad_value)
            padded_sequence = np.concatenate((sequence, pad_seq),
                                             axis=0)
            padded_example[col] = padded_sequence
        return padded_example

    @overload
    def collate(self, batch: Batch) -> Dict[str, List[Any]]:
        batch_len = self.validate_batch(batch)

        padded_batch = {key: [] for key in self.collate_columns}
        for example in HFBasedDataset.batch_samples(batch):
            for col in self.collate_columns:
                sequence = np.array(example[col])
                item_shape = sequence.shape[1:]
                pad_seq = np.full((batch_len - len(sequence), *item_shape),
                                  self.pad_value)
                padded_sequence = np.concatenate((sequence, pad_seq),
                                                 axis=0)
                padded_batch[col].append(padded_sequence)
        for col in self.collate_columns:
            padded_batch[col] = np.stack(padded_batch[col], axis=0)
            padded_batch[col] = torch.tensor(padded_batch[col])
        return padded_batch

    @overload
    def collate(self, batch: list) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        batch : List[Dict[str, Any]]
        """
        batch_len, keys, columns = self.validate_batch(batch)

        padded_batch = {key: [] for key in keys}
        for example in batch:
            for key, value in example.items():
                if key in columns:
                    sequence = np.array(value)
                    item_shape = sequence.shape[1:]
                    pad_seq = np.full((batch_len - len(sequence), *item_shape),
                                      self.pad_value)
                    padded_sequence = np.concatenate((sequence, pad_seq),
                                                     axis=0)
                    padded_batch[key].append(padded_sequence)
                else:
                    padded_batch[key].append(value)
        for col in columns:
            padded_batch[col] = np.stack(padded_batch[col], axis=0)
            padded_batch[col] = torch.tensor(padded_batch[col])
        return padded_batch

    @overload
    def validate_batch(self, batch: Batch) -> int:
        """
        Returns
        -------
        int
            Length that batch should be padded to.
        """
        batch_len = 0 if self.padding_type == 'longest' else self.max_len
        for example in HFBasedDataset.batch_samples(batch):
            length = len(example[self.collate_columns[0]])
            assert all(len(example[col]) == length for col in self.collate_columns)
            if self.max_len is not None and length > self.max_len:
                raise ValueError('at least 1 sequence in batch has length '
                                 'greater than max_len')
            if self.padding_type == 'longest':
                batch_len = max(batch_len, length)
        return batch_len

    @overload
    def validate_batch(self, batch: list) -> Tuple[int, Set[str], Set[str]]:
        """
        Parameters
        ----------
        batch : List[Dict[str, Any]]
        Returns
        -------
        int
            Length that batch should be padded to.
        set
            Batch columns.
        set
            Batch columns that should be padded.
        """
        batch_len = 0 if self.padding_type == 'longest' else self.max_len
        keys = batch[0].keys()
        columns = set(keys) if self.collate_columns is None else self.collate_columns
        for example in batch:
            assert example.keys() == keys, example.keys()
            length = len(example[columns[0]])
            assert all(len(example[col]) == length for col in columns)
            if self.max_len is not None and length > self.max_len:
                raise ValueError('at least 1 sequence in batch has length '
                                 'greater than max_len')
            if self.padding_type == 'longest':
                batch_len = max(batch_len, length)
        return batch_len, set(keys), set(columns)


class SiameesePaddingCollator(PaddingCollator):
    def collate(self, batch: tuple) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        batch : List[Dict[str, Any]]
        """
        batch1 = []
        batch2 = []
        batch3 = []
        for sample1, sample2, sample3 in batch:
            batch1.append(sample1)
            batch2.append(sample2)
            batch3.append(sample3)
        batch1 = super().collate(batch1)
        batch2 = super().collate(batch2)
        batch3 = super().collate(batch3)

        return batch1, batch2, batch3
