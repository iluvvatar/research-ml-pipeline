import torch
from torch import nn
from typing import List

from mlpipeline.models.model import Model
from .units import ConvStack, NerHead
from ....utils import PathLike


class ConvRNN(Model):
    def __init__(self, *,
                 n_ent_types: int,
                 n_classes: int,
                 cnn_layers: int,
                 cnn_kernels: List[int],
                 hid_size: int,
                 rnn: nn.Module,
                 head_layers: int,
                 dropout: float = 0):
        super().__init__()
        # embeddings will be loaded from DeepPavlov/rubert-base-cased
        self.n_ent_types = n_ent_types
        self.n_classes = n_classes
        assert cnn_layers > 0
        emb_size = 768

        # num_embeddings=119547 due to transformers.BertModel vocabulary
        self.embeddings = nn.Embedding(num_embeddings=119547,
                                       embedding_dim=emb_size,
                                       padding_idx=0)
        cnns = [ConvStack(emb_size, emb_size, cnn_kernels, dropout=dropout)]
        cnns += [ConvStack(emb_size, hid_size, cnn_kernels, dropout=dropout) for _ in range(cnn_layers - 1)]
        self.conv = nn.Sequential(*cnns)
        self.rnn = rnn
        self.ner_heads = nn.ModuleList()
        for _ in range(n_ent_types):
            # head = nn.Sequential(
            #     nn.Dropout(dropout),
            #     nn.Linear(in_features=hid_size, out_features=n_classes)
            # )
            head = NerHead(in_channels=hid_size, out_channels=n_classes,
                           n_layers=head_layers, dropout=dropout)
            self.ner_heads.append(head)

    def forward(self, *,
                tokens_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        hidden_states = self.embeddings(tokens_ids)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.permute(2, 0, 1)
        hidden_states = self.rnn(hidden_states)[0]
        hidden_states = hidden_states.permute(1, 0, 2)

        mask = attention_mask\
            .unsqueeze(-1)\
            .repeat_interleave(self.n_classes, dim=-1)
        output = []
        for head in self.ner_heads:
            output.append(head(hidden_states) * mask)
        output = torch.stack(output)
        return output.permute(1, 2, 0, 3)

    def load_pretrained_embeddings(self, embeddings_state_dict_path: PathLike):
        self.embeddings.load_state_dict(torch.load(embeddings_state_dict_path))

    def freeze_embeddings(self, requires_grad=False):
        for param in self.embeddings.parameters():
            param.requires_grad = requires_grad
