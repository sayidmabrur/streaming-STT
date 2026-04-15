import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from train.config import Config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return x


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        x_rot = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (x_rot * sin)


class ASRModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_model = config.embedding_dim

        self.src_proj = nn.Linear(128, self.d_model)

        self.pos_encoder = RotaryPositionalEmbedding(self.d_model, config.max_seq_len)

        # self.pos_encoder = PositionalEncoding(self.d_model, max_len=config.max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        self.output_proj = nn.Linear(self.d_model, config.vocab_size)

    def forward(self, x):
        # Transpose from (batch, n_mels, time) to (batch, time, n_mels)
        x = x.transpose(1, 2)
        x = self.src_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        output = self.output_proj(x)
        return output
