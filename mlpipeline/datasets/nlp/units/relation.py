from dataclasses import dataclass

from ...exceptions import BratFormatError


@dataclass
class Relation:
    """
    Relation extraction task object.
    """
    id: int
    type: str
    arg1_id: int
    arg2_id: int

    def to_brat(self):
        return 'R{}\t{} Arg1:T{} Arg2:T{}'.format(
            self.id,
            self.type,
            self.arg1_id,
            self.arg2_id,
        )

    @classmethod
    def from_brat(cls, line):
        try:
            id_, relation = line.strip().split('\t')
            type_, arg1, arg2 = relation.split()
            return cls(id=int(id_[1:]),
                       type=type_,
                       arg1_id=int(arg1[6:]),
                       arg2_id=int(arg2[6:]))
        except Exception:
            raise BratFormatError(line)

    def __eq__(self, other):
        return self.arg1_id == other.arg1_id and \
               self.arg2_id == other.arg2_id and \
               self.type == other.type
