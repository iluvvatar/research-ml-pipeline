from abc import ABC, abstractmethod
from datasets.arrow_dataset import Batch, Example
from datasets import ClassLabel
from collections import defaultdict
from typing import Union, Iterable, Dict, Any, List
from multimethod import overload

from ....datasets.nlp.units import Entity
from ....datasets import HFBasedDataset
from ...processor import Processor


class Labelizer(Processor):
    """
    Labelizer matches each token to corresponding class label based on
    sentence entities.
    Should be used only for annotated data.
    Must be used after tokenization.
    """
    def __init__(self, *,
                 text_column: str,
                 tokens_column: str,
                 tokens_spans_column: str,
                 entities_column: str,
                 out_labels_column: str = None,
                 out_labels_ids_column: str,
                 predicted_labels_ids_column: str = None,
                 out_predicted_entities_column: str = None,
                 entity_types: List[str],
                 special_tokens: List[str] = (),
                 entities_deserialize_fn=Entity.from_brat,
                 entities_serialize_fn=Entity.to_brat):
        """
        Parameters
        ----------
        text_column : str
            Column name that contains original text that was tokenized.
        tokens_column : str
            Column name that contains tokens.
        tokens_spans_column : str
            Column name that contains tokens spans.
        entities_column : str
            Column name that contains list of entities in brat format
            (see rnnvsbert.datasets.units.Entity for more details about format).
            If not specified, entities will not be filtered for each sentence.
        out_labels_column : str, optional
            Column name in result dataset that will contain text labels for
            each token.
            If not specified labelizer will not create a column with text
            labels.
        out_labels_ids_column : str, optional
            Column name in result dataset that will contain
            numerical representation of labels for each token.
        predicted_labels_ids_column : str
            Column name that will contain list of labels ids after making
            model's prediction.
        out_predicted_entities_column : str, optional
            Column name in result dataset that will contain
            predicted entities in brat format after postprocessing model's
            logits prediction.
        entity_types : List[str]
            List of all possible entity types.
        special_tokens : List[str], optional
            [CLS], [PAD], [UNK] etc.
        """
        self.text_column = text_column
        self.tokens_column = tokens_column
        self.tokens_spans_column = tokens_spans_column
        self.entities_column = entities_column
        self.predicted_labels_ids_column = predicted_labels_ids_column
        self.out_labels_column = out_labels_column
        self.out_labels_ids_column = out_labels_ids_column
        self.out_predicted_entities_column = out_predicted_entities_column

        self.int2ent_type = entity_types
        self.ent_type2int = {
            ent_type: i
            for i, ent_type in enumerate(self.int2ent_type)
        }

        self.special_tokens = set(special_tokens)

        self.entities_deserialize_fn = entities_deserialize_fn
        self.entities_serialize_fn = entities_serialize_fn

    def preprocess(self, dataset: HFBasedDataset, batched=False,
                   use_cached=True, *args, **kwargs) -> HFBasedDataset:
        return dataset.map(self.labelize, batched=batched,
                           load_from_cache_file=use_cached,
                           *args, **kwargs)

    def postprocess(self, dataset: HFBasedDataset, batched=False,
                    use_cached=True, *args, **kwargs) -> HFBasedDataset:
        return dataset.map(self.decode, batched=batched,
                           load_from_cache_file=use_cached,
                           *args, **kwargs)

    @abstractmethod
    @overload
    def labelize(self, batch: Batch) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    @overload
    def labelize(self, example: Example) -> Dict[str, Any]:
        raise NotImplementedError

    @overload
    def decode(self,
               batch: dict,
               batched: bool) -> Union[Dict[str, List[Any]],
                                       Dict[str, Any]]:
        if batched:
            return self.decode(Batch(batch))
        else:
            return self.decode(Example(batch))

    @abstractmethod
    @overload
    def decode(self, batch: Batch) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    @overload
    def decode(self, example: Example) -> Dict[str, Any]:
        raise NotImplementedError

    def filter_intersecting_entities_with_same_type(self,
            entities: Iterable[str], reverse_sort: bool = False
    ) -> Dict[str, List[Entity]]:
        entities = sorted([self.entities_deserialize_fn(e) for e in entities],
                          key=lambda e: e.start)
        cur_entities = {}
        filtered_entities = defaultdict(list)
        for entity in entities:
            if entity.type in cur_entities:
                prev_entity = cur_entities[entity.type]
                if entity.start < prev_entity.stop:
                    if len(prev_entity.spans) > 1 and len(entity.spans) == 1:
                        cur_entities[entity.type] = entity
                    elif len(prev_entity.spans) == 1 and len(entity.spans) > 1:
                        continue
                    elif entity.cover_length > prev_entity.cover_length:
                        cur_entities[entity.type] = entity
                else:
                    filtered_entities[entity.type].append(prev_entity)
                    cur_entities[entity.type] = entity
            else:
                cur_entities[entity.type] = entity
        for _, entity in cur_entities.items():
            filtered_entities[entity.type].append(entity)

        for ent_type, ent_list in filtered_entities.items():
            ent_list.sort(key=lambda e: e.start, reverse=reverse_sort)
        return filtered_entities

    @staticmethod
    def token_and_entity_intersect(entity, token_start, token_stop):
        return any(not (token_stop <= start or stop <= token_start)
                   for start, stop in entity.spans)
