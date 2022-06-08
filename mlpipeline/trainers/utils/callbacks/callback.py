from abc import abstractmethod, ABC


class Callback(ABC):
    @abstractmethod
    def __call__(self, trainer):
        raise NotImplementedError
