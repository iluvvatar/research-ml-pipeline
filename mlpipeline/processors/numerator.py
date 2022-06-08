import numpy as np
from typing import List
from datasets import DatasetDict

from .processor import Processor
from ..datasets import HFBasedDataset


class Numerator(Processor):
    """
    Adds to dataset column with serial number.
    """
    def __init__(self, out_id_column: str = None):
        self.out_id_column = f'<{self.__class__.__name__}>-id' \
            if out_id_column is None else out_id_column

    def preprocess(self,
                   dataset: HFBasedDataset,
                   use_cached=True,
                   *args, **kwargs) -> HFBasedDataset:
        if isinstance(dataset.data, DatasetDict):
            column = {split: np.arange(len(ds))
                      for split, ds in dataset.items()}
        else:
            column = np.arange(len(dataset))
        return dataset.add_column(self.out_id_column, column)

    def postprocess(self,
                    dataset: HFBasedDataset,
                    use_cached=True,
                    *args, **kwargs) -> HFBasedDataset:
        return dataset.remove_columns([self.out_id_column])
