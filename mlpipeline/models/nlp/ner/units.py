import torch
from torch import nn
from typing import List


class ConvStack(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: List[int],
                 dropout: float = 0):
        super().__init__()
        assert out_channels % len(kernel_sizes) == 0
        conv_out_channels = out_channels // len(kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels, out_channels=conv_out_channels,
                      kernel_size=(ks,), padding='same')
            for ks in kernel_sizes
        ])
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x.shape = (N, C, L)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x


class Permute(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.permute(*self.shape)


class NerHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_layers: int,
                 dropout: float = 0):
        assert n_layers > 0
        super().__init__()
        layers = []
        for _ in range(n_layers - 1):
            layer = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_channels, in_channels),
                nn.ReLU(),
                Permute(0, 2, 1),
                nn.BatchNorm1d(num_features=in_channels),
                Permute(0, 2, 1)
            )
            layers.append(layer)
        layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, out_channels)
        )
        layers.append(layer)
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.head(x)
