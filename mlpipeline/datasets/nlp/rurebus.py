from pathlib import Path
from typing import Union, List, Tuple, Iterable
from dataclasses import dataclass
import datasets

from ..dataset import HFBasedDataset
from ...utils import PathLike


HF_PATH = 'iluvvatar/RuREBus'


@dataclass(repr=False)
class RuREBus(HFBasedDataset):
    data: Union[datasets.Dataset, datasets.DatasetDict]
    _ent_types: datasets.Dataset
    _rel_types: datasets.Dataset

    @classmethod
    def load_from_disk(cls, path: PathLike) -> 'RuREBus':
        path = Path(path)
        data = datasets.load_from_disk(str(path / 'data'))
        _ent_types = datasets.load_from_disk(str(path / 'ent_types')).sort('type')
        _rel_types = datasets.load_from_disk(str(path / 'rel_types')).sort('type')
        return cls(data, _ent_types, _rel_types)

    @classmethod
    def load_from_hub(cls, raw_txt=False) -> 'RuREBus':
        if raw_txt:
            data = datasets.load_dataset(HF_PATH, 'raw_txt')
        else:
            data = datasets.load_dataset(HF_PATH)
        _ent_types = datasets.load_dataset(HF_PATH, 'ent_types')
        _rel_types = datasets.load_dataset(HF_PATH, 'rel_types')
        _ent_types = _ent_types['ent_types'].sort('type')
        _rel_types = _rel_types['rel_types'].sort('type')
        return cls(data, _ent_types, _rel_types)

    def save(self, path: PathLike):
        path = Path(path)
        data_path = (path / 'data').absolute()
        ent_types_path = (path / 'ent_types').absolute()
        rel_types_path = (path / 'rel_types').absolute()
        if not data_path.exists():
            data_path.mkdir(parents=True)
        if not ent_types_path.exists():
            ent_types_path.mkdir(parents=True)
        if not rel_types_path.exists():
            rel_types_path.mkdir(parents=True)
        self.data.save_to_disk(str(data_path))
        self._ent_types.save_to_disk(str(ent_types_path))
        self._rel_types.save_to_disk(str(rel_types_path))

    def save_brat(self, path: PathLike, entities_column: str):
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        with open(path / 'ent_types.txt', 'w', encoding='utf-8') as f:
            print('\n'.join(self.entity_types), file=f)
        with open(path / 'rel_types.txt', 'w', encoding='utf-8') as f:
            print('\n'.join(self.relation_types), file=f)
        for example in self.data:
            file_name = f'{example["id"]}.ann'
            with open(path / file_name, 'w', encoding='utf-8') as f:
                print('\n'.join(example[entities_column]), file=f)

    @property
    def splits(self) -> Iterable[str]:
        if not isinstance(self.data, datasets.DatasetDict):
            return []
        return list(self.data.keys())

    @property
    def entity_types(self) -> List[str]:
        return list(self._ent_types['type'])

    @property
    def relation_types(self) -> List[str]:
        return list(self._rel_types['type'])

    @property
    def column_names(self):
        return self.data.column_names
