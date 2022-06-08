from .labelizer import Labelizer


class BIOLabelizer(Labelizer):
    """
    BIO, where instead of I-<class> and along with B-<class> tokens
    marked as <class>

    Example:
         BIO: O, B-person,          I-person,   I-person,   O
         BO:  O, B-person + person, person,     person,     O
    """
    raise NotImplementedError
