from abc import abstractmethod, ABC
from typing import Union, Iterable, List

from ..datasets import Dataset


class Processor(ABC):
    @abstractmethod
    def preprocess(self, dataset: Dataset) -> Dataset:
        """
        Used before feeding into the model.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, dataset: Dataset) -> Dataset:
        """
        Used after processing by the model.
        """
        raise NotImplementedError


class ProcessorsPipeline(Processor):
    def __init__(self,
                 preprocessors: List[Processor],
                 postprocessors: List[Processor] = None):
        self.preprocessors = preprocessors
        self.postprocessors = postprocessors \
            if postprocessors is not None \
            else preprocessors[::-1]

    def preprocess(self, dataset: Dataset) -> Dataset:
        for proc in self.preprocessors:
            dataset = proc.preprocess(dataset)
        return dataset

    def postprocess(self, dataset: Dataset) -> Dataset:
        for proc in self.postprocessors:
            dataset = proc.postprocess(dataset)
        return dataset
