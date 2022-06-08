from pathlib import Path
from typing import List

from ..units import Entity, Relation, Link
from ...exceptions import BratFormatError
from ....utils import PathLike


class BratDoc:
    """
    BratDoc loads brat documents from .ann and .txt files
    and may be converted to json format.
    """
    def __init__(self,
                 text: str,
                 entities: List[Entity],
                 relations: List[Relation],
                 links: List[Link]):
        self.text = text
        self.entities = entities
        self.relations = relations
        self.links = links

    def save_brat(self, path: PathLike,
                  save_text: bool = True,
                  save_entites: bool = True,
                  save_relations: bool = True,
                  save_links: bool = True):
        """
        Parameters
        ----------
        path : pathlib.Path
            path to .ann file to save to
        """
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        if save_text and self.text is not None:
            with open(path.with_suffix('.txt'), 'w', encoding='utf-8') as f_txt:
                f_txt.write(self.text)

        with open(path, 'w', encoding='utf-8') as f_ann:
            if save_entites:
                for entity in self.entities:
                    print(entity.to_brat(), file=f_ann)
            if save_relations:
                for relation in self.relations:
                    print(relation.to_brat(), file=f_ann)
            if save_links:
                for link in self.links:
                    print(link.to_brat(), file=f_ann)

    @classmethod
    def load_brat(cls, path: PathLike):
        """
        Parameters
        ----------
        path : pathlib.Path or str
            path to .ann file to load from
        """
        path = Path(path)
        with open(path.with_suffix('.txt'), encoding="utf-8") as f_txt:
            text = f_txt.read()

        entities = []
        relations = []
        links = []
        with open(path, encoding='utf-8') as f_ann:
            for line in f_ann:
                if line.startswith('T'):
                    entities.append(Entity.from_brat(line))
                elif line.startswith('R'):
                    relations.append(Relation.from_brat(line))
                elif line.startswith('N'):
                    links.append(Link.from_brat(line))
                else:
                    raise BratFormatError(line)
        return cls(text, entities, relations, links)

    def to_dict(self):
        return {
            'text': self.text,
            'entities': [entity.to_brat() for entity in self.entities],
            'relations': [relation.to_brat() for relation in self.relations],
            'links': [link.to_brat() for link in self.links],
        }
