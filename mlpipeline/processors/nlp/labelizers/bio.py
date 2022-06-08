from collections import defaultdict
from typing import Union, Iterable, Dict, Any, List
from datasets.arrow_dataset import Batch, Example
from multimethod import overload
import numpy as np
import logging

from .labelizer import Labelizer
from ....datasets.nlp.units import Entity


class BIOLabelizer(Labelizer):
    int2bio = ['O', 'B', 'I']
    bio2int = {s: i for i, s in enumerate(int2bio)}

    @overload
    def labelize(self, batch: Batch) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @overload
    def labelize(self, example: Example) -> Dict[str, Any]:
        text = example[self.text_column]
        tokens = example[self.tokens_column]
        spans = example[self.tokens_spans_column]
        assert len(tokens) == len(spans), f'tokens={tokens}\nspans={spans}'

        entities = self.filter_intersecting_entities_with_same_type(
            example[self.entities_column],
            reverse_sort=True
        )
        # {entity_type: [current_entity, is_inside]}
        # where is_inside=True means that we have already labeled
        # one token as B (begining) for this entity
        cur_entities = {
            ent_type: [ent_list.pop(), False]
            for ent_type, ent_list in entities.items()
        }

        prev_stop = 0
        labels = []
        labels_ids = []
        for token_idx, (token, (start, stop)) in enumerate(zip(tokens, spans)):
            if start < prev_stop:
                raise ValueError(f'tokens is not sorted '
                                 f'(prev_stop={prev_stop}, start={start})')
            prev_stop = stop

            label = ['O'] * len(self.ent_type2int)
            label_ids = [0] * len(self.ent_type2int)
            for ent_type, i in self.ent_type2int.items():
                if ent_type in cur_entities:
                    while cur_entities[ent_type][0].stop <= start:
                        if entities[ent_type]:
                            cur_entities[ent_type] = [entities[ent_type].pop(), False]
                        else:
                            break
                    if not cur_entities[ent_type]:
                        continue

                    entity, inside = cur_entities[ent_type]
                    if self.token_and_entity_intersect(entity, start, stop):
                        if token.startswith('##'):
                            token = token[2:]
                        # if token not in self.special_tokens and token not in entity.text:
                        #     msg = f'Entity {entity} in text ' \
                        #           f'"...{text[entity.start-1:entity.stop+1]}..." ' \
                        #           f'doesn\'t contain entire token ' \
                        #           f'"{token}" ({start}, {stop}). '\
                        #           f'Token will be labeled as {entity.type}.'
                        #     logging.warning(msg)
                        if inside:
                            j = self.bio2int['I']
                        else:
                            j = self.bio2int['B']
                            cur_entities[ent_type][1] = True
                    else:
                        j = self.bio2int['O']
                else:
                    j = self.bio2int['O']
                label_ids[i] = j
                label[i] = self.int2bio[j]
            labels_ids.append(label_ids)
            labels.append(label)

        assert len(tokens) == len(labels_ids) == len(labels)
        for x, y in zip(labels_ids, labels):
            assert len(self.int2ent_type) == len(x) == len(y)

        if self.out_labels_column is None:
            return {self.out_labels_ids_column: labels_ids}
        else:
            return {self.out_labels_column: labels,
                    self.out_labels_ids_column: labels_ids}

    @overload
    def decode(self, batch: Batch) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @overload
    def decode(self, example: Example) -> Dict[str, Any]:
        raise NotImplementedError
