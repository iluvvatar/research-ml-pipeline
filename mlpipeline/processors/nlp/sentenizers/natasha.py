from razdel import sentenize
from typing import Dict, Any, List
from multimethod import overload
from datasets.arrow_dataset import Batch

from .sentenizer import Sentenizer
from ....datasets.nlp.units import Entity
from ....datasets import HFBasedDataset


class NatashaSentenizer(Sentenizer):
    """
    Sentenizer based on razdel.sentenize()
    """
    @overload
    def sentenize(self, batch: Batch) -> Dict[str, List[Any]]:
        texts = []
        starts = []
        stops = []
        entities = []
        doc_ids = []
        paragraph_split_by = '\n\n'
        for example in HFBasedDataset.batch_samples(batch):
            paragraph_start = 0
            text = example[self.text_column]
            for paragraph in text.split(paragraph_split_by):
                if paragraph.strip():
                    for sent in sentenize(paragraph):
                        sent_start = paragraph_start + sent.start
                        sent_stop = paragraph_start + sent.stop
                        assert sent.text == text[sent_start:sent_stop], sent

                        texts.append(sent.text)
                        starts.append(sent_start)
                        stops.append(sent_stop)

                        if self.entities_column is not None:
                            sent_entities = self.sentence_entities(
                                example[self.entities_column],
                                sent.text, sent_start, sent_stop
                            )
                            entities.append(sent_entities)
                        if self.doc_id_column is not None:
                            doc_ids.append(example[self.doc_id_column])

                paragraph_start += len(paragraph) + len(paragraph_split_by)
        result = {self.out_text_column: texts,
                  self.out_start_column: starts,
                  self.out_stop_column: stops}
        if self.out_entities_column is not None:
            result[self.out_entities_column] = entities
        if self.out_doc_id_column is not None:
            result[self.out_doc_id_column] = doc_ids
        return result

    @overload
    def sentenize(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

    def sentence_entities(self, entities: List[str], sent_text, start: int, stop: int):
        entities = [self.entities_deserialize_fn(e) for e in entities]
        entities = [e for e in filter(
            lambda e: start < e.stop and e.start < stop,
            entities
        )]
        for e in entities:
            for span in e.spans:
                span[0] = max(0, span[0] - start)
                span[1] = min(stop, span[1] - start)
                # assert sent_text[span[0]:span[1]]
                # assert sent_text[span[0]:span[1]] in e.text, f'"{sent_text[span[0]:span[1]]}" in "{e.text}"'
        return [self.entities_serialize_fn(e) for e in entities]

