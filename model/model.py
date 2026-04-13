from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    vocab_size: int = 1000
    hidden_size: int = 300
    embedding_dim: int = 768
    num_layers: int = 3
    batch_size: int = 16
    max_seq_len: int = 2048


config = Config()


class OutputLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # Since we are using a Bidirectional LSTM, the output feature dimension is hidden_size * 2
        self.out = nn.Linear(config.hidden_size * 2, config.vocab_size)

    def forward(self, x):
        x = self.out(x)
        return x


class RNNEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.input_proj = nn.Linear(128, config.embedding_dim)

        self.encoder = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if config.num_layers > 1 else 0.0,
        )

    def forward(self, x):
        x = self.input_proj(x)
        # LSTM returns (output, (h_n, c_n)), we only need the output sequence
        x, _ = self.encoder(x)
        return x


class ASRModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.encoder = RNNEncoder(config)
        self.output = OutputLayer(config)

    def forward(self, x):
        # Transpose from (batch, n_mels, time) to (batch, time, n_mels)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = self.output(x)

        return x
