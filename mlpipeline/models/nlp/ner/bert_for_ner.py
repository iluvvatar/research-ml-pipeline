from transformers import BertModel
import torch
from torch import nn
from typing import Union

from mlpipeline.models.model import Model
from .units import NerHead
from ....utils import PathLike


class BertForNER(Model):
    def __init__(self,
                 bert_name_or_path: Union[str, PathLike],
                 n_ent_types: int,
                 n_classes: int,
                 dropout: float = 0,
                 n_head_layers: int = 2,
                 return_hidden_states: bool = False):
        super().__init__()
        self.return_hidden_states = return_hidden_states

        self.bert = BertModel.from_pretrained(bert_name_or_path,
                                              add_pooling_layer=False)
        self.ner_heads = nn.ModuleList()
        for _ in range(n_ent_types):
            # head = nn.Sequential(
            #     nn.Dropout(0.1),
            #     nn.Linear(in_features=self.bert.config.hidden_size,
            #               out_features=n_classes),
            #
            # )
            head = NerHead(in_channels=768, out_channels=n_classes,
                           n_layers=n_head_layers, dropout=dropout)
            self.ner_heads.append(head)

    def forward(self, *,
                tokens_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        hidden_states = self.bert(
            input_ids=tokens_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        output = []
        for head in self.ner_heads:
            output.append(head(hidden_states))
        output = torch.stack(output).permute(1, 2, 0, 3)
        mask = attention_mask\
            .unsqueeze(-1)\
            .repeat_interleave(output.shape[-2], dim=-1)\
            .unsqueeze(-1)\
            .repeat_interleave(output.shape[-1], dim=-1)
        output = output * mask
        if self.return_hidden_states:
            return output, hidden_states
        else:
            return output

    def freeze_embeddings(self, requires_grad=False):
        for param in self.bert.get_input_embeddings().parameters():
            param.requires_grad = requires_grad
