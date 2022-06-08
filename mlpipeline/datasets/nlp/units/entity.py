from dataclasses import dataclass
from typing import List

from ...exceptions import BratFormatError, FormatError


@dataclass
class Entity:
    """
    Named entity recognition task object.
    """
    id: int
    type: str
    spans: List[List[int]]
    text: str

    @classmethod
    def from_brat(cls, line: str):
        """
        Parameters
        ----------
        line : str
            Brat format entity string.
            Brat format entity: "<ID>\t<TYPE> <SPANS>\t<TEXT>", where <SPANS>
            may consists of any number of "<START> <STOP>" joined by ";",
            because entity may consists of several separate parts.
            <ID> - entity id starting with "T", e.g. T42
            <TYPE> - entity type, e.g. ORGANIZATION
            <START> - first symbol position of entity part (int), e.g. 152
            <STOP> - last symbol position of entity part + 1 (int), e.g. 159
            <TEXT> - full text of an entity, e.g. Sabaton

        Returns
        -------
        Entity

        Raises
        ------
        BratFormatError
            If format if incorrect.
        """
        try:
            id_, entity, text = line.split('\t', 2)
            type_, spans = entity.split(maxsplit=1)
            spans = [s.split() for s in spans.split(';')]
            spans = [[int(start), int(stop)] for start, stop in spans]
            return cls(id=int(id_[1:]),
                       type=type_,
                       spans=spans,
                       text=text)
        except Exception as exc:
            raise BratFormatError(f'{line}\nOriginal exception: {type(exc)} {exc}')

    def to_brat(self):
        return 'T{}\t{} {}\t{}'.format(
            self.id,
            self.type,
            ';'.join([f'{start} {stop}' for start, stop in self.spans]),
            self.text,
        )

    @classmethod
    def from_str(cls, line: str):
        try:
            start, stop, type_ = line.split()
            start = int(start)
            stop = int(stop)
            return cls(id=0, type=type_, spans=[[start, stop]], text='')
        except Exception as exc:
            raise FormatError(f'{line}\nOriginal exception: {type(exc)} {exc}')

    def to_str(self):
        return f'{self.start} {self.stop} {self.type}'

    def to_tuple(self):
        return self.start, self.stop, self.type

    @property
    def start(self):
        return self.spans[0][0]

    @property
    def stop(self):
        return self.spans[-1][1]

    @property
    def cover_length(self):
        return sum(stop - start for start, stop in self.spans)

    def __eq__(self, other):
        return self.spans == other.spans and self.type == other.type
