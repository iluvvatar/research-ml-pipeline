from abc import ABC, abstractmethod
from typing import Union, Dict, Any, List
from datasets.arrow_dataset import Batch, Example
from multimethod import overload

from ...processor import Processor
from ....datasets import HFBasedDataset


class Tokenizer(Processor):
    """
    Tokenizers split texts into tokens and their numerical representation.
    __call__ works as in huggingface tokenizers.
    Must be used after sentenization.
    """
    def __init__(self, *,
                 text_column: str,
                 out_tokens_column: str = None,
                 out_spans_column: str,
                 out_tokens_ids_column: str):
        """
        Parameters
        ----------
        text_column : str
            Column name that contains text to be tokenized.
        out_tokens_column : str, optional
            Column name in result dataset that will contain text tokens.
            If not specified tokenizer will not create a column with text
            labels.
        out_spans_column : str
            Column name in result dataset that will contain spans for tokens,
            i.e. tuple of (start, stop) for each token.
        out_tokens_ids_column : str
            Column name in result dataset that will contain numerical
            representation of tokens.
        """

        self.text_column = text_column

        self.out_tokens_column = out_tokens_column
        self.out_spans_column = out_spans_column
        self.out_tokens_ids_column = out_tokens_ids_column

    def preprocess(self,
                   dataset: HFBasedDataset,
                   use_cached=True,
                   *args, **kwargs) -> HFBasedDataset:
        dataset = dataset.map(self.tokenize, batched=False,
                              load_from_cache_file=use_cached,
                              *args, **kwargs)
        return dataset

    def postprocess(self,
                    dataset: HFBasedDataset,
                    use_cached=True,
                    *args, **kwargs) -> HFBasedDataset:
        raise NotImplementedError

    @abstractmethod
    @overload
    def tokenize(self, batch: Batch) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    @overload
    def tokenize(self, example: Example) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    @overload
    def tokenize(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def special_tokens(self) -> List[str]:
        raise NotImplementedError
