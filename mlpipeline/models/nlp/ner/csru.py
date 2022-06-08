import sru

from .crnn import ConvRNN


class ConvSRU(ConvRNN):
    def __init__(self, *,
                 rnn_layers: int,
                 hid_size: int,
                 dropout: float,
                 **kwargs):
        rnn = sru.SRU(input_size=hid_size,
                      hidden_size=hid_size // 2,
                      num_layers=rnn_layers,
                      bidirectional=True,
                      dropout=dropout)
        super().__init__(hid_size=hid_size, dropout=dropout, rnn=rnn, **kwargs)
