from pathlib import Path
from typing import Iterable, Union
from dataclasses import dataclass
import datasets

from ..dataset import HFBasedDataset
from ...utils import PathLike
from .clean_text import clean_text


HF_PATH = 'IlyaGusev/gazeta'


@dataclass(repr=False)
class Gazeta(HFBasedDataset):
    """
    https://github.com/IlyaGusev/gazeta
    """
    data: Union[datasets.Dataset, datasets.DatasetDict]

    @classmethod
    def load_from_disk(cls, path: PathLike) -> 'Gazeta':
        data = datasets.load_from_disk(path)
        return cls(data)

    @classmethod
    def load_from_hub(cls) -> 'Gazeta':
        data = datasets.load_dataset(HF_PATH)
        data = data.map(lambda ex: {'text': clean_text(ex['text'])})
        for split in data:
            data[split].info.supervised_keys = None
        return cls(data)

    def save(self, path: PathLike):
        path = Path(path).absolute()
        if not path.exists():
            path.mkdir(parents=True)
        self.data.save_to_disk(str(path))

    @property
    def splits(self) -> Iterable[str]:
        return 'train', 'test', 'validation'

    @property
    def column_names(self):
        return self.data.column_names
