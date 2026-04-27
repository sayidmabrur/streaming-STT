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


# class AttentionBlock

class MultiHeadAttention(nn.Module):
    def __init__(self, config, dropout=0.01):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = config.num_heads
        self.head_dim = config.n_mels // config.num_heads
        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.sdpa =False

    def forward(self, x):
        B, T, C = x.shape
        head_dim = C // self.num_heads

        Q = x.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        K = x.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        V = x.view(B, T, self.num_heads, head_dim).transpose(1, 2)

        if self.sdpa:
            y = 0
            pass
        else:
            attn_score = (Q @ K.transpose(-2, -1)) * (math.sqrt(K.shape[-1]))
            y = attn_score @ V

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(y)
        out = self.dropout(out)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout, n_mels):
        super(ScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):

        # B, T, C = x.shape
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        q_k = Q @ K.transpose(-2, -1)
        attn_score = torch.softmax((q_k / (self.d_model**0.5)), dim=-1)
        out = attn_score @ V
        out = self.out_proj(out)
        return out

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
    def __init__(self, config):
        super(EncoderLayer, self).__init__()

        # Referenced from: OpenAI's Whisper implementation
        # They're using Conv1d to project mel spectrogram into embedding dimension
        self.input_proj = nn.Conv1d(config.n_mels, config.embedding_dim, kernel_size=3, padding=1)
        self.input_proj2 = nn.Conv1d(config.embedding_dim, config.embedding_dim, kernel_size=3, padding=1)

        self.config = config
        self.relu = nn.ReLU()

        self.attention = MultiHeadAttention(config)

    def forward(self, x):

        # x = x.transpose(1,2)
        x = self.input_proj(x)
        x = self.relu(x)
        x = self.input_proj2(x)
        x = self.relu(x)
        # print("input_proj:", x.shape)

        # x = x.transpose(1,2)
        x = self.attention(x)

        # print(x.shape)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        # self.pos_encoder = PositionalEncoding(config.d_model, config.block_size)

    def forward(self, x):
        x = self.pos_encoder(x)
        return x



class QuasTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_model = config.embedding_dim

        self.audio_encoder = EncoderLayer(
            config,
        )


        self.text_decoder = DecoderLayer(
            config,
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.audio_encoder(x)

        return x


from train.config import model_cfg

BLOCK_SIZE = 4096
EMBEDDING_DIM = model_cfg.embedding_dim
N_MELS = 128

train_sample = torch.randn(8, N_MELS, BLOCK_SIZE)
model = QuasTransformer(model_cfg)
x = model(train_sample)
# print(x.shape)
# print(model)
