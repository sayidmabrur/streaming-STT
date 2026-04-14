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


# implementation of RoPE a copy-pasta from : https://medium.com/ai-insights-cobet/rotary-positional-embeddings-a-detailed-look-and-comprehensive-understanding-4ff66a874d83
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()

        # Create a rotation matrix.
        self.rotation_matrix = torch.zeros(
            d_model, d_model, device=torch.device("cuda")
        )

        i = torch.arange(0, d_model, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(0, d_model, dtype=torch.float32).unsqueeze(0)

        angles = i * j * 0.01
        self.rotation_matrix = torch.cos(angles)
        # for i in range(d_model):
        #     for j in range(d_model):
        #         self.rotation_matrix[i, j] = torch.cos(i * j * 0.01)

        # Create a positional embedding matrix.
        self.positional_embedding = torch.zeros(
            max_seq_len, d_model, device=torch.device("cuda")
        )
        i = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(0, d_model, dtype=torch.float32).unsqueeze(0)

        self.positional_embedding = torch.cos(i * j * 0.01)
        # print(self.positional_embedding.shape)
        # for i in range(max_seq_len):
        #     for j in range(d_model):
        #         self.positional_embedding[i, j] = torch.cos(i * j * 0.01)

    def forward(self, x):
        """
        Args:
            x: A tensor of shape (batch_size, seq_len, d_model).

        Returns:
            A tensor of shape (batch_size, seq_len, d_model).
        """

        # Add the positional embedding to the input tensor.
        # We slice the positional embedding to match the sequence length of x,
        # just like the original PositionalEncoding did with x.size(1)
        x = x + self.positional_embedding[: x.size(1), :].to(x.device)

        # Apply the rotation matrix to the input tensor.
        x = torch.matmul(x, self.rotation_matrix.to(x.device))

        return x


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
