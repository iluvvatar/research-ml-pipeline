from typing import List, Dict, Any
from datasets.arrow_dataset import Batch, Example
from transformers import BertTokenizer
import razdel
from multimethod import overload

from .tokenizer import Tokenizer
from ....utils import PathLike


class RuBERTTokenizer(Tokenizer):
    def __init__(self, *,
                 pretrained_tokenizer_path: PathLike = None,
                 out_word_tokens_indices_column: str = None,
                 out_attention_mask_column: str = None,
                 out_token_type_ids_column: str = None,
                 **kwargs):
        """
        Parameters
        ----------
        bert_tokenizer_path : str, optional
            Path to pretrained transformers.BertTokenizer.
            If not specified, will try to download DeepPavlov/rubert-base-cased
            from Hugging Face Hub.
        out_word_tokens_indices_column : str, optional
            Column name in result dataset that will contain word tokens
            indices, i.e. tuple of (start_token_idx, stop_token_idx), where
            start_token_idx is the first token index in returned tokens and
            stop_token_idx is the last token index in returned tokens + 1.
            Word tokens indices are used in Viterbi prediction decoding
            algorithm.
            If not specified tokenizer will not create a column with word
            tokens indices.
        out_attention_mask_column : str, optional
            Column name in result dataset that will contain attention mask
            for tokens for BERT model.
            If not specified tokenizer will not create a column with attention
            masks.
        out_token_type_ids_column : str, optional
            Column name in result dataset that will contain token type ids for
            BERT model.
            If not specified tokenizer will not create a column with
            token type ids.
        """
        super().__init__(**kwargs)
        if pretrained_tokenizer_path is None:
            pretrained_tokenizer_path = 'DeepPavlov/rubert-base-cased'
        self._tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer_path)
        self.out_word_tokens_indices_column = out_word_tokens_indices_column
        self.out_attention_mask_column = out_attention_mask_column
        self.out_token_type_ids_column = out_token_type_ids_column

    @overload
    def tokenize(self, batch: Batch) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @overload
    def tokenize(self, example: Example) -> Dict[str, Any]:
        text = example[self.text_column]
        # word tokenization by razdel
        razdel_spans = []
        razdel_tokens = []
        for token in razdel.tokenize(text):
            razdel_tokens.append(token.text)
            razdel_spans.append((token.start, token.stop))

        # tokenize each word into subwords by BertTokenizer
        tokens = [self._tokenizer.cls_token]
        spans = [(0, 0)]    # for [CLS] token
        cur_token_idx = 1
        word_tokens_indices = []
        for span, token in zip(razdel_spans, razdel_tokens):
            word_start = span[0]
            stop = 0
            subtokens = self._tokenizer.tokenize(token)
            if self._tokenizer.unk_token in subtokens:
                word_tokens_indices.append((cur_token_idx, cur_token_idx + 1))
                cur_token_idx += 1
                tokens.append(self._tokenizer.unk_token)
                spans.append(span)
            else:
                word_tokens_indices.append(
                    (cur_token_idx, cur_token_idx + len(subtokens))
                )
                cur_token_idx += len(subtokens)
                for subtoken in subtokens:
                    tokens.append(subtoken)
                    if subtoken.startswith('##'):
                        subtoken = subtoken[2:]
                    start = token.find(subtoken, stop)
                    if start == -1:
                        raise ValueError(f'"{subtoken}" not found in "{token}"')
                    stop = start + len(subtoken)
                    assert subtoken == text[word_start + start:word_start + stop]
                    spans.append((word_start + start, word_start + stop))
        tokens.append(self._tokenizer.sep_token)
        l = len(text)
        spans.append((l, l))    # for [SEP] token

        tokens_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(tokens)
        token_type_ids = [0] * len(tokens)
        assert len(tokens) == len(spans) == len(tokens_ids) \
               == len(attention_mask) == len(token_type_ids)

        result = {self.out_spans_column: spans,
                  self.out_tokens_ids_column: tokens_ids}
        if self.out_tokens_column is not None:
            result[self.out_tokens_column] = tokens
        if self.out_attention_mask_column is not None:
            result[self.out_attention_mask_column] = attention_mask
        if self.out_token_type_ids_column is not None:
            result[self.out_token_type_ids_column] = token_type_ids
        if self.out_word_tokens_indices_column is not None:
            result[self.out_word_tokens_indices_column] = word_tokens_indices
        return result

    @overload
    def tokenize(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

    @overload
    def __call__(self, batch: Batch) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def special_tokens(self) -> List[str]:
        return self._tokenizer.all_special_tokens
