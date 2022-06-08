from collections import defaultdict
from typing import Union, Iterable, Dict, Any, List
from datasets.arrow_dataset import Batch, Example
from multimethod import overload
import numpy as np
import logging
import torch

from .labelizer import Labelizer
from ....datasets.nlp.units import Entity
from ....datasets import HFBasedDataset


class BILOULabelizer(Labelizer):
    int2bilou = ['O', 'B', 'L', 'I', 'U']
    bilou2int = {s: i for i, s in enumerate(int2bilou)}

    # Used in Viterbi algorithm
    first_subword_transition_probs = np.array([
        [1/2, 1/2,   0,   0,   0],  # O -> O B
        [  0,   0,   0,   1,   0],  # B -> I
        [1/2, 1/2,   0,   0,   0],  # L -> O B
        [  0,   0,   0,   1,   0],  # I -> I
        [1/2, 1/2,   0,   0,   0]   # U -> O B
    ], np.float64)
    middle_subword_transition_probs = np.array([
        [  1,   0,   0,   0,   0],  # O -> O
        [  0,   0,   0,   1,   0],  # B -> I
        [  0,   0,   0,   0,   0],  # L -> no possible
        [  0,   0,   0,   1,   0],  # I -> I
        [  0,   0,   0,   0,   0]   # U -> no possible
    ], np.float64)
    last_subword_transition_probs = np.array([
        [  1,   0,   0,   0,   0],  # O -> O
        [  0,   0, 1/2, 1/2,   0],  # B -> L I
        [  0,   0,   0,   0,   0],  # L -> no possible
        [  0,   0, 1/2, 1/2,   0],  # I -> L I
        [  0,   0,   0,   0,   0]   # U -> no possible
    ], np.float64)
    word_transition_probs = np.array([
        [1/3, 1/3,   0,   0, 1/3],  # O -> O B U
        [  0,   0, 1/2, 1/2,   0],  # B -> L I
        [1/3, 1/3,   0,   0, 1/3],  # L -> O B U
        [  0,   0, 1/2, 1/2,   0],  # I -> L I
        [1/3, 1/3,   0,   0, 1/3]   # U -> O B U
    ], np.float64)

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
                            if (token_idx < len(example['tokens']) - 1 and
                                    self.token_and_entity_intersect(entity, *(example['spans'][token_idx+1]))):
                                j = self.bilou2int['I']
                            else:
                                j = self.bilou2int['L']
                        else:
                            if (token_idx < len(example['tokens']) - 1 and
                                    self.token_and_entity_intersect(entity, *(example['spans'][token_idx+1]))):
                                j = self.bilou2int['B']
                                cur_entities[ent_type][1] = True
                            else:
                                j = self.bilou2int['U']
                    else:
                        j = self.bilou2int['O']
                else:
                    j = self.bilou2int['O']
                label_ids[i] = j
                label[i] = self.int2bilou[j]
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
    def decode(self,
               batch: dict,
               batched: bool) -> Union[Dict[str, List[Any]],
                                       Dict[str, Any]]:
        if batched:
            return self.decode(Batch(batch))
        else:
            return self.decode(Example(batch))

    @overload
    def decode(self, batch: Batch) -> Dict[str, List[Any]]:
        decoded = defaultdict(list)
        for example in HFBasedDataset.batch_samples(batch):
            for key, value in self.decode(Example(example)).items():
                decoded[key].append(value)
        return dict(decoded)

    @overload
    def decode(self, example: Example) -> Dict[str, Any]:
        text = example[self.text_column]
        spans = example[self.tokens_spans_column]
        labels_ids = np.array(example[self.predicted_labels_ids_column]).transpose()
        pred_entities = []
        ent_id = 0
        for ent_type_id, labels_ids_for_ent_type in enumerate(labels_ids):
            ent_type = self.int2ent_type[ent_type_id]
            entity_start = None
            assert len(labels_ids_for_ent_type) == len(spans)
            for i, (label_id, span) in enumerate(zip(labels_ids_for_ent_type, spans)):
                label = self.int2bilou[label_id]
                if label == 'U':
                    if entity_start is not None:
                        raise ValueError(f'Got U when previous entity having '
                                         f'no end. Pos {i}: '
                                         f'{labels_ids_for_ent_type}')
                    pred_entities.append(Entity(id=ent_id,
                                                type=ent_type,
                                                spans=[span],
                                                text=text[span[0]:span[1]]))
                    ent_id += 1
                elif label == 'B':
                    if entity_start is not None:
                        raise ValueError(f'Got B when previous entity having '
                                         f'no end. Pos {i}: '
                                         f'{labels_ids_for_ent_type}')
                    entity_start = span[0]
                elif label == 'I':
                    if entity_start is None:
                        raise ValueError(f'Got I when entity was not '
                                         f'started yet. Pos {i}: '
                                         f'{labels_ids_for_ent_type}')
                elif label == 'L':
                    if entity_start is None:
                        raise ValueError(f'Got L when entity was not '
                                         f'started yet. Pos {i}: '
                                         f'{labels_ids_for_ent_type}')
                    entity_stop = span[1]
                    pred_entities.append(Entity(id=ent_id,
                                                type=ent_type,
                                                spans=[[entity_start,
                                                        entity_stop]],
                                                text=text[entity_start:
                                                          entity_stop]))
                    ent_id += 1
                    entity_start = None
                else:
                    if entity_start is not None and span[0] != span[1]:
                        raise ValueError(f'Got O when previous entity having '
                                         f'no end. Pos {i}: '
                                         f'{labels_ids_for_ent_type}')
            if entity_start is not None:
                entity_stop = spans[-1][1]
                pred_entities.append(Entity(id=ent_id,
                                            type=ent_type,
                                            spans=[[entity_start,
                                                    entity_stop]],
                                            text=text[entity_start:
                                                      entity_stop]))
        pred_entities = [self.entities_serialize_fn(e) for e in pred_entities]
        return {self.out_predicted_entities_column: pred_entities}

    @overload
    def decode_prediction(self, batch: Batch) -> Dict[str, List[Any]]:
        # TODO
        raise NotImplementedError
