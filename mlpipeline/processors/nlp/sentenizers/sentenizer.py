from abc import abstractmethod
from typing import Union, Dict, Any, List, Callable
from datasets.arrow_dataset import Batch, Example
from multimethod import overload
from collections import defaultdict
import datasets

from ...processor import Processor
from ....datasets import HFBasedDataset
from ....datasets.nlp.units import Entity


class Sentenizer(Processor):
    """
    Sentenizers split Dataset texts and entities into sentences.
    """
    def __init__(self, *,
                 text_column: str,
                 doc_id_column: str,
                 entities_column: str = None,
                 remove_columns: List[str] = None,
                 out_text_column: str,
                 out_start_column: str,
                 out_stop_column: str,
                 out_doc_id_column: str,
                 out_entities_column: str = None,
                 pred_entities_column: str = None,
                 postprocess_remove_columns: List[str] = None,
                 unite_entities: bool = False,
                 entities_deserialize_fn: Callable[[str], Entity] = Entity.from_brat,
                 entities_serialize_fn: Callable[[Entity], str] = Entity.to_brat):
        """
        Parameters
        ----------
        text_column : str
            Column name that contains text to be sentenized.
        doc_id_column : str
            Column name that contains document id, that can be used
            to unite sentences back into document.
        entities_column : str, optional
            Column name that contains list of entities in brat format
            (see rnnvsbert.datasets.units.Entity for more details about format).
            If not specified, entities will not be filtered for each sentence.
        remove_columns : List[str], optional
            Columns to be removed. Should contain all columns
            from (set(out columns) - set(in columns)).
        out_text_column: str
            Column name in result dataset that will contain sentence text.
        out_start_column: str
            Column name in result dataset that will contain sentence
            start position.
        out_stop_column: str
            Column name in result dataset that will contain sentence
            stop position.
        out_doc_id_column: str
            Column name in result dataset that will contain document id
            for sentence.
        out_entities_column: str, optional
            Column name in result dataset that will contain sentence entities.
            Should be specified if entities_column is specified.
        """
        if entities_column is not None:
            assert out_entities_column is not None

        self.text_column = text_column
        self.entities_column = entities_column
        self.doc_id_column = doc_id_column

        self.remove_columns = remove_columns if remove_columns else []

        self.out_text_column = out_text_column
        self.out_start_column = out_start_column
        self.out_stop_column = out_stop_column
        self.out_entities_column = out_entities_column
        self.out_doc_id_column = out_doc_id_column

        self.pred_entities_column = pred_entities_column
        self.unification_buffer = None
        self.number_of_sentences_in_docs = None   # TODO: fix
        self.postprocess_remove_columns = postprocess_remove_columns
        self.unite_entities_ = unite_entities

        self.entities_deserialize_fn = entities_deserialize_fn
        self.entities_serialize_fn = entities_serialize_fn

    def preprocess(self,
                   dataset: HFBasedDataset,
                   use_cached: bool = True,
                   *args, **kwargs) -> HFBasedDataset:
        return dataset.map(self.sentenize, batched=True,
                           load_from_cache_file=use_cached,
                           remove_columns=self.remove_columns,
                           *args, **kwargs)

    def postprocess(self,
                    dataset: HFBasedDataset,
                    use_cached: bool = True,
                    *args, **kwargs) -> HFBasedDataset:
        # TODO: fix number_of_sentences_in_doc - create necessary for
        #  unification output attributes in sentenize
        if isinstance(dataset.data, datasets.DatasetDict):
            raise NotImplementedError('Unification for DatasetDict is not implemented :((( I have to fix this')
        self.unification_buffer = defaultdict(list)
        self.number_of_sentences_in_docs = defaultdict(int)
        for example in dataset:
            self.number_of_sentences_in_docs[example[self.out_doc_id_column]] += 1

        return dataset.map(self.unite, batched=True,
                           load_from_cache_file=use_cached,
                           remove_columns=self.postprocess_remove_columns,
                           *args, **kwargs)

    @abstractmethod
    @overload
    def sentenize(self, batch: Batch) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    @overload
    def sentenize(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

    def unite(self, batch: Union[Batch, Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
        if self.entities_column is not None and self.entities_column == self.out_entities_column:
            raise NotImplementedError('I don\'t know why, but entities_column == out_entities_column don\'t work')
        united_batch = defaultdict(list)
        for sample in HFBasedDataset.batch_samples(batch):
            united_sample = self.update_unification_buffer(sample)
            if united_sample is not None:
                for col, value in united_sample.items():
                    united_batch[col].append(value)
        return dict(united_batch)

    def update_unification_buffer(self, sample: Dict[str, Any]) -> Union[None, Dict[str, Any]]:
        # TODO: fix number_of_sentences_in_doc - create necessary for
        #  unification output attributes in sentenize
        doc_id = sample[self.out_doc_id_column]
        buffer = self.unification_buffer[doc_id]
        buffer.append(sample)
        buffer.sort(key=lambda s: s[self.out_start_column])
        if len(buffer) == self.number_of_sentences_in_docs[doc_id]:
            return self.unite_doc_samples(buffer)
        return None

    def unite_doc_samples(self, samples: List[Dict[str, Any]]):
        united_sample = {}
        united_sample[self.doc_id_column] = samples[0][self.out_doc_id_column]
        united_sample[self.text_column] = self.unite_texts(samples)
        if self.unite_entities_:
            united_sample[self.entities_column] = self.unite_entities(samples)
            for e in united_sample[self.entities_column]:
                e = self.entities_deserialize_fn(e)
                text = united_sample[self.text_column]
                e_text = text[e.start:e.stop]
                assert e.text == e_text, f'{e} {e_text}\n{text}'
        if self.pred_entities_column is not None:
            united_sample[self.pred_entities_column] = self.unite_predicted_entities(samples)
            for e in united_sample[self.pred_entities_column]:
                e = self.entities_deserialize_fn(e)
                text = united_sample[self.text_column]
                e_text = text[e.start:e.stop]
                assert e.text == e_text, f'{e} {e_text}\n{text}'
        # print(united_sample[self.doc_id_column])
        # entities = [Entity.from_brat(e) for e in united_sample['entities']]
        # predicted_entities = [Entity.from_brat(e) for e in united_sample['predicted_entities']]
        # entities.sort(key=lambda e: e.start)
        # predicted_entities.sort(key=lambda e: e.start)
        # print(entities)
        # print(predicted_entities)
        # print('============================================================')
        return united_sample

    def unite_texts(self, samples: List[Dict[str, Any]]):
        text = []
        prev_stop = 0
        for sample in samples:
            start = sample[self.out_start_column]
            assert start >= prev_stop
            text.append(' ' * (start - prev_stop))
            text.append(sample[self.out_text_column])
            prev_stop = sample[self.out_stop_column]
        return ''.join(text)

    def unite_entities(self, samples: List[Dict[str, Any]]):
        # TODO: add sentence_id column and sentences_count
        #       and fix unification accordingly
        entities = []
        entities_buffer = {}
        for sample in samples:
            prev_entities_buffer = entities_buffer
            entities_buffer = {}
            sent_start = sample[self.out_start_column]
            sent_stop = sample[self.out_stop_column]
            for e in sample[self.out_entities_column]:
                e = self.entities_deserialize_fn(e)
                if len(e.spans) > 1:
                    raise NotImplementedError(
                        'entities separated on several parts can not be '
                        'processed in Sentenizer.uite() :((( '
                        'I have to fix it')
                for span in e.spans:
                    span[0] += sent_start
                    span[1] += sent_start
                if e.start == sent_start:
                    e_prev_part = prev_entities_buffer.get(e.type, None)
                    if (e_prev_part is not None
                            and e.text == e_prev_part.text
                            and (e_prev_part.stop - e_prev_part.start) + (e.stop - e.start) == len(e.text)):
                        e.spans[0][0] = e_prev_part.spans[0][0]
                if e.stop == sent_stop:
                    assert e.type not in entities_buffer
                    entities_buffer[e.type] = e
                else:
                    entities.append(self.entities_serialize_fn(e))
            while prev_entities_buffer:
                _, e = prev_entities_buffer.popitem()
                entities.append(self.entities_serialize_fn(e))
        while entities_buffer:
            _, e = entities_buffer.popitem()
            entities.append(self.entities_serialize_fn(e))
        return entities

    def unite_predicted_entities(self, samples: List[Dict[str, Any]]):
        pred_entities = []
        for sample in samples:
            sent_start = sample[self.out_start_column]
            # if sent_start == 0:
            #     print(sample[self.out_doc_id_column])
            #     entities = [Entity.from_brat(e) for e in sample['entities']]
            #     predicted_entities = [Entity.from_brat(e) for e in sample['predicted_entities']]
            #     entities.sort(key=lambda e: e.start)
            #     predicted_entities.sort(key=lambda e: e.start)
            #     print(entities)
            #     print(predicted_entities)
            #     print('============================================================')
            for e in sample[self.pred_entities_column]:
                e = self.entities_deserialize_fn(e)
                for span in e.spans:
                    # if span[0] == 0 and sent_start == 0:
                    #     print(f'{span} + {sent_start}', end=' -> ')
                    span[0] += sent_start
                    span[1] += sent_start
                    # if span[0] == 0:
                    #     print(span, e)
                pred_entities.append(self.entities_serialize_fn(e))
        return pred_entities
