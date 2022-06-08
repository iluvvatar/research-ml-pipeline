from abc import ABC, abstractmethod
from typing import Callable, Iterator, List, Union, Dict, Any, Iterable, Tuple
import numpy as np
from pathlib import Path
import datasets
from datasets.arrow_dataset import Batch, Example
from dataclasses import dataclass
from multimethod import overload

from torch.utils.data import \
    Dataset as TorchDataset, IterableDataset as TorchIterableDataset

from ..utils import PathLike


class Dataset(ABC):
    @classmethod
    @abstractmethod
    def load(cls, path: PathLike) -> 'Dataset':
        raise NotImplementedError

    @abstractmethod
    def save(self, path: PathLike):
        raise NotImplementedError

    @abstractmethod
    def map(self, function: Callable, batched: bool = False) -> 'Dataset':
        raise NotImplementedError

    @abstractmethod
    def add_column(self, name: str, column: Union[list, np.ndarray]) -> 'Dataset':
        raise NotImplementedError

    @abstractmethod
    def remove_columns(self, names: List[str]) -> 'Dataset':
        raise NotImplementedError

    @abstractmethod
    def rename_column(self, name: str, new_name: str) -> 'Dataset':
        raise NotImplementedError

    @abstractmethod
    def sort(self, column: str, reverse: bool = True, use_cached: bool = True) -> 'Dataset':
        raise NotImplementedError

    @abstractmethod
    def shuffle(self) -> 'Dataset':
        raise NotImplementedError

    @abstractmethod
    def select(self, indices: List[int]) -> 'Dataset':
        raise NotImplementedError

    @abstractmethod
    def split(self) -> List['Dataset']:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def column_names(self):
        raise NotImplementedError


class MapDataset(Dataset, TorchDataset):
    @abstractmethod
    def __getitem__(self, index) -> dict:
        raise NotImplementedError


class IterableDataset(Dataset, TorchIterableDataset):
    @abstractmethod
    def __iter__(self) -> Iterator:
        raise NotImplementedError


@dataclass(repr=False)
class HFBasedDataset(MapDataset):
    data: Union[datasets.DatasetDict, datasets.Dataset]

    @classmethod
    def load(cls, path: PathLike = None) -> 'HFBasedDataset':
        if path is None:
            return cls.load_from_hub()
        path = Path(path)
        return cls.load_from_disk(path)

    @classmethod
    @abstractmethod
    def load_from_disk(cls, path: PathLike) -> 'HFBasedDataset':
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_from_hub(cls) -> 'HFBasedDataset':
        raise NotImplementedError

    @property
    @abstractmethod
    def splits(self) -> Iterable[str]:
        raise NotImplementedError

    def map(self, *args, **kwargs) -> 'HFBasedDataset':
        data = self.data.map(*args, **kwargs)
        return self.__class__(data=data,
                              **self._exclude_key(self.__dict__, 'data'))

    def filter(self, *args, **kwargs) -> 'HFBasedDataset':
        data = self.data.filter(*args, **kwargs)
        return self.__class__(data=data,
                              **self._exclude_key(self.__dict__, 'data'))

    @overload
    def add_column(self, name: str, columns: dict):
        """
        Usable when isinstance(self.data, datasets.DatasetDict).

        Parameters
        ----------
        name : str
            Column name
        columns : Dict[str, Union[list, np.ndarray]]
            Dict where keys are split names and values are columns for the this
            splits.

        Returns
        -------
        HFBasedDataset
        """
        if isinstance(self.data, datasets.Dataset):
            raise ValueError('Use add_column(name, column) for split dataset')
        assert columns.keys() == self.data.keys()
        data = datasets.DatasetDict({
            split: ds.add_column(name, columns[split])
            for split, ds in self.data.items()
        })
        return self.__class__(data=data,
                              **self._exclude_key(self.__dict__, 'data'))

    @overload
    def add_column(self, name: str, column: np.ndarray):
        """
        Usable when isinstance(self.data, datasets.Dataset).

        Parameters
        ----------
        name : str
            Column name
        column : Union[list, np.ndarray]
            Column values.

        Returns
        -------
        HFBasedDataset
        """
        if isinstance(self.data, datasets.DatasetDict):
            raise ValueError(
                'Use add_column(name, columns) for unsplit dataset'
            )
        data = self.data.add_column(name, column)
        return self.__class__(data=data,
                              **self._exclude_key(self.__dict__, 'data'))

    @overload
    def add_column(self, name: str, column: list):
        return self.add_column(name, np.array(column))

    def rename_column(self, name: str, new_name: str) -> 'HFBasedDataset':
        data = self.data.rename_column(original_column_name=name,
                                            new_column_name=new_name)
        return self.__class__(data=data,
                              **self._exclude_key(self.__dict__, 'data'))

    def remove_columns(self, columns: List[str]) -> 'HFBasedDataset':
        data = self.data.remove_columns(columns)
        return self.__class__(data=data,
                              **self._exclude_key(self.__dict__, 'data'))

    def sort(self, column, reverse=False, use_cached=True) -> 'HFBasedDataset':
        data = self.data.sort(column=column, reverse=reverse,
                                   load_from_cache_file=use_cached)
        return self.__class__(data=data,
                              **self._exclude_key(self.__dict__, 'data'))

    def shuffle(self) -> 'HFBasedDataset':
        data = self.data.shuffle()
        return self.__class__(data=data,
                              **self._exclude_key(self.__dict__, 'data'))

    def select(self, indices: Union[List[int], np.ndarray]) -> 'HFBasedDataset':
        if isinstance(self.data, datasets.DatasetDict):
            raise ValueError('Split the dataset before')
        data = self.data.select(indices)
        return self.__class__(data=data,
                              **self._exclude_key(self.__dict__, 'data'))

    def split(self) -> Iterable['HFBasedDataset']:
        if not isinstance(self.data, datasets.DatasetDict):
            raise ValueError('Dataset has been already split')
        return (self.__class__(self.data[split],
                               **self._exclude_key(self.__dict__, 'data'))
                for split in self.splits)

    def items(self) -> Iterable[Tuple[str, 'HFBasedDataset']]:
        if not isinstance(self.data, datasets.DatasetDict):
            raise ValueError('Dataset has been already split')
        for split, ds in self.data.items():
            yield split, self.__class__(
                data=self.data[split], **self._exclude_key(self.__dict__, 'data')
            )

    def __getitem__(self, index) -> Union[Example, 'HFBasedDataset']:
        if isinstance(self.data, datasets.DatasetDict):
            data = self.data[index]
            return self.__class__(data=data,
                                  **self._exclude_key(self.__dict__, 'data'))
        return self.data[index]

    def __iter__(self) -> Iterator:
        return self.data.__iter__()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return self.data.__repr__()

    @staticmethod
    def batch_samples(batch: Union[Batch, Dict[str, List[Any]]]) -> List[Dict[str, Any]]:
        batch_size = len(batch[next(iter(batch))])
        for i in range(batch_size):
            sample = {}
            for column in batch:
                sample[column] = batch[column][i]
            yield sample

    @staticmethod
    def _exclude_key(dictionary, key) -> Dict[str, Any]:
        return {k: v for k, v in dictionary.items() if k != key}
