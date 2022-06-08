from pathlib import Path
from typing import Union, List, Tuple, Iterable
from dataclasses import dataclass
import datasets

from ..dataset import HFBasedDataset
from ...utils import PathLike


HF_PATH = 'iluvvatar/RuNNE'


@dataclass(repr=False)
class RuNNE(HFBasedDataset):
    """
    https://github.com/dialogue-evaluation/RuNNE
    """
    data: Union[datasets.Dataset, datasets.DatasetDict]
    _ent_types: datasets.Dataset

    @classmethod
    def load_from_disk(cls, path: PathLike) -> 'RuNNE':
        path = Path(path)
        data = datasets.load_from_disk(str(path / 'data'))
        _ent_types = datasets.load_from_disk(str(path / 'ent_types')).sort('type')
        return cls(data, _ent_types)

    @classmethod
    def load_from_hub(cls) -> 'RuNNE':
        data = datasets.load_dataset(HF_PATH)
        _ent_types = datasets.load_dataset(HF_PATH, 'ent_types')
        _ent_types = _ent_types['ent_types'].sort('type')
        return cls(data, _ent_types)

    def save(self, path: PathLike):
        path = Path(path)
        data_path = (path / 'data').absolute()
        ent_types_path = (path / 'ent_types').absolute()
        if not data_path.exists():
            data_path.mkdir(parents=True)
        if not ent_types_path.exists():
            ent_types_path.mkdir(parents=True)
        self.data.save_to_disk(str(data_path))
        self._ent_types.save_to_disk(str(ent_types_path))

    @property
    def splits(self) -> Iterable[str]:
        return 'train', 'test', 'dev'

    @property
    def entity_types(self) -> List[str]:
        return list(self._ent_types['type'])

    @property
    def column_names(self):
        return self.data.column_names
