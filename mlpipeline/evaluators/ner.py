from typing import Dict, Any, List, Callable
import datasets
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from time import time

from ..datasets import HFBasedDataset
from ..datasets.nlp.units import Entity
from ..models import Model
from .evaluator import Evaluator

from tqdm import tqdm


class EvaluatorNER(Evaluator):
    def __init__(self,
                 real_entities_column: str,
                 pred_entities_column: str,
                 entity_types: List[str],
                 entities_deserialize_fn=Entity.from_brat,
                 few_shot_types: List[str] = ()):
        self.real_entities_column = real_entities_column
        self.pred_entities_column = pred_entities_column

        self.entities_deserialize_fn = entities_deserialize_fn

        self.entity_types = entity_types
        self.few_shot_types = few_shot_types

    def evaluate(self, dataset: HFBasedDataset) -> Dict[str, Any]:
        assert isinstance(dataset.data, datasets.Dataset)
        n_entity_types = len(self.entity_types)
        tps = [0] * n_entity_types
        fps = [0] * n_entity_types
        fns = [0] * n_entity_types
        for example in tqdm(dataset, desc='Evaluation'):
            real_entities = example[self.real_entities_column]
            pred_entities = example[self.pred_entities_column]
            real_entities = [self.entities_deserialize_fn(e) for e in real_entities]
            pred_entities = [self.entities_deserialize_fn(e) for e in pred_entities]
            real_entities = [(e.type, e.start, e.stop) for e in real_entities]
            pred_entities = [(e.type, e.start, e.stop) for e in pred_entities]
            for ent_type_id, ent_type in enumerate(self.entity_types):
                real = set(filter(lambda e: e[0] == ent_type, real_entities))
                pred = set(filter(lambda e: e[0] == ent_type, pred_entities))
                tps[ent_type_id] += len(real & pred)
                fps[ent_type_id] += len(pred - real)
                fns[ent_type_id] += len(real - pred)
                # if pred - real or real - pred:
                #     print(pred - real)
                #     print(real - pred)
                #     pred_texts = [example['text'][start:stop] for _, start, stop in pred]
                #     real_texts = [example['text'][start:stop] for _, start, stop in real]
                #     print('pred =', pred, pred_texts)
                #     print('real =', real, real_texts)
                #     print(len(example['tokens']))
                #     print(example['tokens'])
                #     labels_ids = [l[ent_type_id] for l in example['labels_ids']]
                #     predicted_labels_ids = [l[ent_type_id] for l in example['predicted_labels_ids']]
                #     print('predicted_labels_ids =', predicted_labels_ids)
                #     print('labels_ids =', labels_ids)
                #     print('=================================')
        precisions = [tp / (tp + fp) if (tp + fp) != 0 else 0 for tp, fp in zip(tps, fps)]
        recalls = [tp / (tp + fn) if (tp + fn) != 0 else 0 for tp, fn in zip(tps, fns)]
        f1_scores = [2 * p * r / (p + r) if (p + r) != 0 else 0 for p, r in zip(precisions, recalls)]
        result = {ent_type: f1_scores[i] for i, ent_type in enumerate(self.entity_types)}
        result['f1-macro'] = sum(f1_scores) / len(f1_scores)
        if self.few_shot_types:
            f1_main = [f1 for i, f1 in enumerate(f1_scores)
                       if self.entity_types[i] not in self.few_shot_types]
            f1_few_shot = [f1 for i, f1 in enumerate(f1_scores)
                           if self.entity_types[i] in self.few_shot_types]
            result['f1-main'] = sum(f1_main) / len(f1_main)
            result['f1-few-shot'] = sum(f1_few_shot) / len(f1_few_shot)
        return result


class NerRnnTimeEvaluator(Evaluator):
    def __init__(self, *,
                 batch_size: int,
                 collate_fn: Callable,
                 tokens_ids_column: str,
                 attention_mask_column: str,
                 verbose: bool = True):
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        self.tokens_ids_column = tokens_ids_column
        self.attention_mask_column = attention_mask_column

        self.verbose = verbose

    def evaluate(self,
                 model: Model,
                 dataset: HFBasedDataset,
                 devices: List[str] = ('cpu', 'cuda')) -> Dict[str, Any]:
        # костылями едиными
        model.eval()
        model.freeze_embeddings()
        result = {}

        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            collate_fn=self.collate_fn)
        with torch.no_grad():
            for device in devices:
                model.to(device)
                total_time = 0
                for batch in tqdm(loader, desc=device, disable=not self.verbose):
                    tokens_ids = batch[self.tokens_ids_column].to(device)
                    attention_mask = batch[self.attention_mask_column].to(device)
                    start_time = time()
                    logits = model(tokens_ids=tokens_ids, attention_mask=attention_mask)
                    total_time += time() - start_time
                result[f'{device}-total-time'] = total_time
                result[f'{device}-time-per-batch'] = total_time / len(loader)

        return result
