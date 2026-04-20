# type: ignore

import math

import torch
import torch.nn as nn

from train.config import Config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return x


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, block_size):
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


# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads

#     def forward(self, x):
#         pass

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(ScaledDotProductAttention, self).__init__()
        # self.num_heads = num_heads
        # self.head_dim = d_model // num_heads

        self.w_q = nn.Parameter(torch.randn(d_model, d_model))
        self.w_k = nn.Parameter(torch.randn(d_model, d_model))
        self.w_v = nn.Parameter(torch.randn(d_model, d_model))
        self.out_proj = nn.Parameter(torch.randn(d_model, d_model))


    def forward(self, x):

        Q = x @ self.w_q
        K = x @ self.w_k
        V = x @ self.w_v

        attn_score=(Q @ K.T) / torch.sqrt(Q.shape[-1])
        attn_score = torch.softmax(attn_score, dim=-1)

        return attn_score
        # out = attention @ V
        # out = out @ self.out_proj



# class EfficientAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super(EfficientAttention, self).__init__()


#         self.rope = RotaryPositionalEmbedding(d_model, block_size=2048)

#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads

#         self.q = nn.Parameter(torch.randn(d_model, d_model))
#         # self.k = nn.Parameter(torch.randn(d_model, d_model))
#         # self.v = nn.Parameter(torch.randn(d_model, d_model))
#         # self.out_proj = nn.Parameter(torch.randn(d_model, d_model))

#         # self.q = nn.Linear(d_model, d_model)
#         # self.k = nn.Linear(d_model, d_model)
#         # self.v = nn.Linear(d_model, d_model)
#         # self.out_proj = nn.Linear(d_model, d_model)

#         # self.d_model = d_model
#         # self.num_heads =num_heads
#         # self.head_dim = d_model //num_heads

#         # self.q_proj = nn.Linear(d_model, d_model)
#         # self.k_proj = nn.Linear(d_model, d_model)
#         # self.v_proj = nn.Linear(d_model, d_model)
#         # self.out_proj = nn.Linear(d_model, d_model)

#     def forward(self, x):
#         pass


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, block_size, num_heads, dropout):
        super(EncoderLayer, self).__init__()

        self.attention = EfficientAttention(
            d_model, num_heads, block_size
        )
        # self.pos_encoder = PositionalEncoding(d_model, block_size)

    def forward(self, x):
        return x
        # x = self.pos_encoder(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, block_size):
        super(DecoderLayer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, block_size)

    def forward(self, x):
        x = self.pos_encoder(x)
        return x


class QuasTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_model = config.embedding_dim

        self.src_proj = nn.Linear(128, self.d_model)

        # self.pos_encoder = RotaryPositionalEmbedding(self.d_model, config.block_size)
        # self.pos_encoder = PositionalEncoding(self.d_model, config.block_size)
        self.output_proj = nn.Linear(self.d_model, config.vocab_size)

        self.transformer_encoder = EncoderLayer(
            self.d_model,
            config.block_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.transformer_decoder = DecoderLayer(
            self.d_model,
            config.block_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

    def forward(self, x):
        # Transpose from (batch, n_mels, time) to (batch, time, n_mels)
        x = x.transpose(1, 2)
        x = self.src_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.transformer_decoder(x)
        output = self.output_proj(x)
        return output
