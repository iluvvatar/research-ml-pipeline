from dataclasses import dataclass

from ...exceptions import BratFormatError


@dataclass
class Link:
    """
    Entity linking task object.
    """
    id: int
    entity_id: int
    reference: str
    text: str

    def to_brat(self):
        return 'N{}\tReference T{} {}\t{}'.format(
            self.id,
            self.entity_id,
            self.reference,
            self.text,
        )

    @classmethod
    def from_brat(cls, line):
        try:
            id_, link, text = line.strip().split('\t')
            _, entity_id, reference = link.split()
            return cls(id=int(id_[1:]),
                       entity_id=int(entity_id[1:]),
                       reference=reference,
                       text=text)
        except Exception:
            raise BratFormatError(line)

    def __eq__(self, other):
        return self.entity_id == other.entity_id and \
               self.reference == other.reference
