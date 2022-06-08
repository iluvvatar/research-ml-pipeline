import torch
from typing import Dict, Any, Tuple

from .trainer import Trainer
from ..processors.nlp.prediction_postprocessors import Viterbi
from ..processors.nlp.labelizers import Labelizer
from .utils.losses import LossList, TripleSiameeseCosineDistanceLoss


class NerTripleSiameeseTrainer(Trainer):
    def __init__(self, *,
                 loss_fn: LossList,
                 tokens_ids_column: str,
                 attention_mask_column: str,
                 labels_ids_column: str,
                 siam_entity_mask_column: str,
                 siam_entity_type_id_column: str,
                 logits_column: str,
                 hidden_states_column: str,
                 viterbi_decoder: Viterbi,
                 labelizer: Labelizer,
                 **kwargs):
        assert isinstance(loss_fn[0], TripleSiameeseCosineDistanceLoss)
        super().__init__(loss_fn=loss_fn, **kwargs)
        self.tokens_ids_column = tokens_ids_column
        self.attention_mask_column = attention_mask_column
        self.labels_ids_column = labels_ids_column
        self.siam_entity_mask_column = siam_entity_mask_column
        self.siam_entity_type_id_column = siam_entity_type_id_column
        self.logits_column = logits_column
        self.hidden_states_column = hidden_states_column

        self.viterbi = viterbi_decoder
        self.labelizer = labelizer

    def train_step(self, triple_batch: Tuple[Dict[str, Any], ...]) -> float:
        self.optimizer.zero_grad()
        for batch in triple_batch:
            tokens_ids = batch[self.tokens_ids_column].to(self.device)
            attention_mask = batch[self.attention_mask_column].to(self.device)
            logits, hidden_states = self.model(
                tokens_ids=tokens_ids, attention_mask=attention_mask
            )
            batch[self.logits_column] = logits.permute(2, 0, 3, 1)
            batch[self.hidden_states_column] = hidden_states
            batch[self.labels_ids_column] = batch[self.labels_ids_column]\
                .long()\
                .permute(2, 0, 1)\
                .to(self.device)
            batch[self.siam_entity_mask_column] = batch[self.siam_entity_mask_column]\
                .to(self.device)

        # Loss ================================================================
        siam_kwargs = {'hidden_states': [batch[self.hidden_states_column] for batch in triple_batch],
                       'masks': [batch[self.siam_entity_mask_column] for batch in triple_batch]}
        ce_kwargs = {'kwargs': [
            {'kwargs': [
                {'input': logits,
                 'target': labels_ids}
                for logits, labels_ids in zip(batch[self.logits_column],
                                              batch[self.labels_ids_column])
            ]} for batch in triple_batch
        ]}
        loss = self.loss_fn(kwargs=[siam_kwargs, ce_kwargs])

        # Optimizer step ======================================================
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def eval_step(self, batch: Dict[str, Any]) -> float:
        tokens_ids = batch[self.tokens_ids_column].to(self.device)
        attention_mask = batch[self.attention_mask_column].to(self.device)
        logits, _ = self.model(tokens_ids=tokens_ids,
                               attention_mask=attention_mask)

        # Metrics =============================================================
        attention_mask = attention_mask.bool()
        logits = [logits_i[mask_i].detach().cpu().numpy()
                  for logits_i, mask_i in zip(logits, attention_mask)]
        batch[self.logits_column] = logits
        batch = self.val_postprocess_batch_fn(batch)
        batch.update(self.viterbi.decode(batch, batched=True))
        batch.update(self.labelizer.decode(batch, batched=True))

        for metric in self.metrics:
            metric.add_batch(batch)

        return 0
