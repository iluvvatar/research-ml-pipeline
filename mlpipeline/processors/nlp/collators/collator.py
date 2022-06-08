from abc import abstractmethod
from typing import Dict, Any, List
from datasets.arrow_dataset import Batch, Example
from multimethod import overload
import torch

from ...processor import Processor
from ....datasets import HFBasedDataset


class Collator(Processor):
    """
    Collators operates over sequence length. May split, truncate, pad
    etc. sequence.
    """
    def __init__(self, *, collate_columns: List[str]):
        """
        Parameters
        ----------
        collate_columns : list[str]
            Columns that should be collated. May be processed differently
            by different collators.
        """
        assert len(collate_columns) > 0
        self.collate_columns = collate_columns

    def preprocess(self,
                   dataset: HFBasedDataset,
                   use_cached=True,
                   *args, **kwargs) -> HFBasedDataset:
        return dataset.map(self.collate, batched=False,
                           load_from_cache_file=use_cached,
                           *args, **kwargs)

    def postprocess(self,
                    dataset: HFBasedDataset,
                    use_cached=True,
                    *args, **kwargs) -> HFBasedDataset:
        raise NotImplementedError

    @abstractmethod
    @overload
    def collate(self, example: Example) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    @overload
    def collate(self, batch: Batch) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    @overload
    def collate(self, batch: list) -> Dict[str, torch.Tensor]:
        """
        Used in torch.utils.data.DataLoader(collate_fn=...)

        Parameters
        ----------
        batch : List[Dict[str, Any]]
        """
        raise NotImplementedError
