import torch
from typing import Dict, Any

from .trainer import Trainer
from ..processors.nlp.prediction_postprocessors import Viterbi
from ..processors.nlp.labelizers import Labelizer
from .utils.losses import LossList


class NerTrainer(Trainer):
    def __init__(self, *,
                 loss_fn: LossList,
                 tokens_ids_column: str,
                 attention_mask_column: str,
                 labels_ids_column: str,
                 logits_column: str,
                 viterbi_decoder: Viterbi,
                 labelizer: Labelizer,
                 **kwargs):
        """
        Parameters
        ----------
        loss_fn : LossList
            Loss function for each entity type.
        tokens_ids_column
        attention_mask_column
        labels_ids_column
        logits_column
        viterbi_decoder
        labelizer
        kwargs
        """
        super().__init__(loss_fn=loss_fn, **kwargs)
        self.tokens_ids_column = tokens_ids_column
        self.attention_mask_column = attention_mask_column
        self.labels_ids_column = labels_ids_column
        self.logits_column = logits_column

        self.viterbi = viterbi_decoder
        self.labelizer = labelizer

    def train_step(self, batch: Dict[str, Any]) -> float:
        self.optimizer.zero_grad()
        tokens_ids = batch[self.tokens_ids_column].to(self.device)
        attention_mask = batch[self.attention_mask_column].to(self.device)
        labels_ids = batch[self.labels_ids_column].long().to(self.device)
        logits = self.model(tokens_ids=tokens_ids,
                               attention_mask=attention_mask)

        # Loss ================================================================
        loss = self.compute_loss(logits, labels_ids)

        # Optimizer step ======================================================
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def eval_step(self, batch: Dict[str, Any]) -> float:
        tokens_ids = batch[self.tokens_ids_column].to(self.device)
        attention_mask = batch[self.attention_mask_column].to(self.device)
        labels_ids = batch[self.labels_ids_column].long().to(self.device)
        logits = self.model(tokens_ids=tokens_ids,
                            attention_mask=attention_mask)

        # Loss ================================================================
        loss = self.compute_loss(logits, labels_ids)

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

        return loss.item()

    def compute_loss(self, logits, labels_ids):
        # B - batch, L - seq, E - ent types, C - classes
        logits = logits.permute(2, 0, 3, 1)         # (B, L, E, C) -> (E, B, C, L)
        labels_ids = labels_ids.permute(2, 0, 1)    # (B, L, E) -> (E, B, L)
        kwargs = [{'input': logits_i, 'target': labels_ids_i}
                  for logits_i, labels_ids_i in zip(logits, labels_ids)]
        return self.loss_fn(kwargs=kwargs)
