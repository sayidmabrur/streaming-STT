# type: ignore

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.head_dim = config.embedding_dim // config.num_heads
        # Learned Q, K, V projections — essential for the model to learn different subspaces
        self.w_q = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.w_k = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.w_v = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.out_proj = nn.Linear(
            config.embedding_dim, config.embedding_dim, bias=False
        )
        self.dropout_val = dropout

        # Zero-init out_proj so each attention block starts as an identity residual.
        # Without this, 8 stacked random projections compound variance and the
        # loss explodes to 8-10 at init (GPT-2 / Megatron-LM trick).
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x):
        B, T, C = x.shape

        # Project into Q, K, V then split into heads
        Q = self.w_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Flash Attention: O(T) memory instead of O(T²) — avoids the huge attn_score matrix
        y = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            dropout_p=self.dropout_val if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


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


class ResidualAttentionBlock(nn.Module):
    def __init__(self, config, cross_attn=False):
        super(ResidualAttentionBlock, self).__init__()

        self.attention = MultiHeadAttention(config)
        self.ln_attn = nn.LayerNorm(config.embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim * 4, config.embedding_dim),
            nn.Dropout(config.dropout),
        )
        self.ln_mlp = nn.LayerNorm(config.embedding_dim)
        self.cross_attn = cross_attn

        nn.init.zeros_(self.mlp[3].weight)
        nn.init.zeros_(self.mlp[3].bias)

    def forward(self, x):
        x = x + self.attention(self.ln_attn(x))
        if self.cross_attn:
            pass
            # x = x + self.attention(x, x)
        x = x + self.mlp(self.ln_mlp(x))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()

        # Referenced from: OpenAI's Whisper implementation
        # They're using Conv1d to project mel spectrogram into embedding dimension
        self.input_proj = nn.Conv1d(
            config.n_mels, config.embedding_dim, kernel_size=3, padding=1
        )
        self.input_proj2 = nn.Conv1d(
            config.embedding_dim, config.embedding_dim, kernel_size=3, padding=1
        )

        self.config = config

        # self.attention = MultiHeadAttention(config)
        # self.attention = ResidualAttentionBlock(config)
        self.attn_blocks = nn.ModuleList(
            [ResidualAttentionBlock(config) for _ in range(config.num_layers)]
        )

        self.post_ln = nn.LayerNorm(config.embedding_dim)

    def forward(self, x):

        # x = x.transpose(1,2)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = F.gelu(x)
        x = self.input_proj2(x)
        x = F.gelu(x)
        x = x.permute(0, 2, 1)
        for block in self.attn_blocks:
            x = block(x)
        x = self.post_ln(x)

        # print(x.shape)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()

        self.pos_embedding = RotaryPositionalEmbedding(
            config.embedding_dim, config.block_size
        )

        # self.attn_blocks =
        # self.pos_encoder = PositionalEncoding(config.d_model, config.block_size)

    def forward(self, x):
        x = self.pos_embedding(x)
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

        self.vocab_proj = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.audio_encoder(x)
        x = self.vocab_proj(x)

        return x


from train.config import model_cfg

BLOCK_SIZE = 4096
EMBEDDING_DIM = model_cfg.embedding_dim
N_MELS = 128

# train_sample = torch.randn(8, N_MELS, BLOCK_SIZE)
# model = QuasTransformer(model_cfg)
# x = model(train_sample)
# print(x.shape)
# print(model)
