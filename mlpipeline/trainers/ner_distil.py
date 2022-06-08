import torch
from typing import Dict, Any

from .trainer import Trainer
from ..processors.nlp.prediction_postprocessors import Viterbi
from ..processors.nlp.labelizers import Labelizer


class NerDistilTrainer(Trainer):
    def __init__(self, *,
                 tokens_ids_column: str,
                 logits_column: str,
                 attention_mask_column: str,
                 viterbi_decoder: Viterbi,
                 labelizer: Labelizer,
                 **kwargs):
        super().__init__(**kwargs)
        self.tokens_ids_column = tokens_ids_column
        self.logits_column = logits_column
        self.attention_mask_column = attention_mask_column

        self.viterbi = viterbi_decoder
        self.labelizer = labelizer

    def train_step(self, batch: Dict[str, Any]) -> float:
        self.optimizer.zero_grad()
        tokens_ids = batch[self.tokens_ids_column].to(self.device)
        attention_mask = batch[self.attention_mask_column].to(self.device)
        logits = batch[self.logits_column].float().to(self.device)
        pred_logits = self.model(tokens_ids=tokens_ids,
                                 attention_mask=attention_mask)
        loss = self.loss_fn(input=pred_logits,
                            target=logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def eval_step(self, batch: Dict[str, Any]) -> float:
        tokens_ids = batch[self.tokens_ids_column].to(self.device)
        attention_mask = batch[self.attention_mask_column].to(self.device)
        logits = self.model(tokens_ids=tokens_ids,
                            attention_mask=attention_mask)
        attention_mask = attention_mask.bool()
        logits = [logits[i][attention_mask[i]].detach().cpu().numpy()
                  for i in range(len(logits))]
        batch[self.logits_column] = logits
        batch.update(self.viterbi.decode(batch, batched=True))
        batch.update(self.labelizer.decode(batch, batched=True))
        for metric in self.metrics:
            metric.add_batch(batch)
        return 0
