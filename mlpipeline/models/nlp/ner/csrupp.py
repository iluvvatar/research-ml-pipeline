import sru

from .crnn import ConvRNN


class ConvSRUpp(ConvRNN):
    def __init__(self, *,
                 rnn_layers: int,
                 hid_size: int,
                 dropout: float,
                 **kwargs):
        rnn = sru.SRUpp(input_size=hid_size,
                        hidden_size=hid_size // 2,
                        proj_size=hid_size // 4,
                        num_layers=rnn_layers,
                        bidirectional=True,
                        dropout=dropout,
                        attn_dropout=dropout)
        super().__init__(hid_size=hid_size, dropout=dropout, rnn=rnn, **kwargs)
