# type: ignore

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.config import Config


# ── Positional encodings ────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Absolute sinusoidal PE — added to the encoder residual stream.

    The encoder uses global (non-causal) attention, so absolute PE is the
    right choice.  It is added BEFORE pre_ln so the pattern survives the
    first LayerNorm (LayerNorm normalises across the feature dim, preserving
    the relative differences that encode position).
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class RotaryPositionalEmbedding(nn.Module):
    """RoPE — must be applied to Q and K AFTER their linear projections,
    INSIDE MultiHeadAttention.  Never apply to the residual stream because
    LayerNorm (applied before attention) would immediately erase the rotation.

    Expects tensors of shape [B, num_heads, T, head_dim].
    """

    def __init__(self, head_dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: [B, H, T, head_dim]
        seq_len = x.size(2)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)                  # [T, head_dim/2]
        emb   = torch.cat((freqs, freqs), dim=-1)              # [T, head_dim]
        cos   = emb.cos()[None, None]                          # [1, 1, T, head_dim]
        sin   = emb.sin()[None, None]
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        x_rot = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (x_rot * sin)


# ── Attention ────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """Single attention module shared by self-attention and cross-attention.

    Parameters
    ----------
    use_rope : bool
        When True, RoPE is applied to Q and K after projection.  Should be
        True for decoder self-attention, False everywhere else (encoder
        self-attention uses sinusoidal PE on the residual stream;
        cross-attention should NOT rotate the encoder K/V).
    """

    def __init__(self, config, use_rope: bool = False, dropout: float = 0.01):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim  = config.embedding_dim // config.num_heads

        self.w_q      = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.w_k      = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.w_v      = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.dropout_val = dropout

        self.rope = RotaryPositionalEmbedding(self.head_dim) if use_rope else None

        # Zero-init out_proj: block starts as identity residual (GPT-2 / Megatron trick).
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x, xa=None, mask=None):
        B, T, C = x.shape
        kv_src = xa if xa is not None else x
        S = kv_src.size(1)

        q = self.w_q(x     ).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(kv_src).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(kv_src).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE rotates q and k in their head-local subspace — position info
        # survives the dot-product and is never erased by LayerNorm.
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_val if self.training else 0.0,
            attn_mask=mask,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


# ── Transformer block ────────────────────────────────────────────────────────

class ResidualAttentionBlock(nn.Module):
    """Pre-norm residual block.  Optionally includes cross-attention.

    Parameters
    ----------
    use_rope : bool
        Forwarded to the self-attention module (decoder blocks set this True).
    cross_attn : bool
        When True, a cross-attention sub-layer is inserted between self-attn
        and FFN.
    """

    def __init__(self, config, cross_attn: bool = False, use_rope: bool = False):
        super().__init__()

        # ── self-attention ──────────────────────────────────────────────────
        self.attn    = MultiHeadAttention(config, use_rope=use_rope)
        self.ln_attn = nn.LayerNorm(config.embedding_dim)

        # ── cross-attention (decoder only) ─────────────────────────────────
        self.cross_attn    = MultiHeadAttention(config, use_rope=False) if cross_attn else None
        self.cross_attn_ln = nn.LayerNorm(config.embedding_dim)         if cross_attn else None
        # Separate LN for the encoder KV so the query and key/value spaces
        # are independently normalised before the cross-attention dot-product.
        self.cross_attn_kv_ln = nn.LayerNorm(config.embedding_dim)      if cross_attn else None

        # ── feed-forward ────────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim * 4, config.embedding_dim),
            nn.Dropout(config.dropout),
        )
        self.ln_mlp = nn.LayerNorm(config.embedding_dim)

        # Zero-init the down-projection so the MLP also starts as identity.
        nn.init.zeros_(self.mlp[3].weight)
        nn.init.zeros_(self.mlp[3].bias)

    def forward(self, x, xa=None, mask=None):
        # Self-attention (causal for decoder, full for encoder)
        x = x + self.attn(self.ln_attn(x), mask=mask)

        # Cross-attention: Q from decoder, K/V from (normalised) encoder output
        if (self.cross_attn is not None
                and self.cross_attn_ln is not None
                and self.cross_attn_kv_ln is not None
                and xa is not None):
            xa_norm = self.cross_attn_kv_ln(xa)
            x = x + self.cross_attn(self.cross_attn_ln(x), xa=xa_norm)

        x = x + self.mlp(self.ln_mlp(x))
        return x


# ── Encoder ──────────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """Audio encoder.

    Normalization flow
    ------------------
    conv → GELU → conv → GELU  (feature extraction)
         → sinusoidal PE        (absolute position)
         → pre_ln               (anchors residual stream before first block)
         → N × ResidualAttentionBlock (each has internal pre-norm)
         → post_ln              (normalises xa before decoder uses it as K/V)
    """

    def __init__(self, config):
        super().__init__()
        self.input_proj  = nn.Conv1d(config.n_mels,       config.embedding_dim, kernel_size=3, padding=1)
        self.input_proj2 = nn.Conv1d(config.embedding_dim, config.embedding_dim, kernel_size=3, padding=1)

        # Absolute PE for encoder (global attention → no causal constraint).
        # max_len must cover MAX_FRAMES (1024) from collate_fn, NOT block_size
        # (which is the decoder token budget and is intentionally much smaller).
        self.pos_encoding = SinusoidalPositionalEncoding(
            config.embedding_dim,
            max_len=config.max_audio_frames,
        )

        # Anchor the residual stream ONCE before the first attention block.
        # Without this, the raw conv output (arbitrary scale) enters the first
        # block and the residual connection can grow unboundedly across layers.
        self.pre_ln = nn.LayerNorm(config.embedding_dim)

        self.attn_blocks = nn.ModuleList(
            [ResidualAttentionBlock(config) for _ in range(config.num_layers)]
        )
        self.post_ln = nn.LayerNorm(config.embedding_dim)

    def forward(self, x):
        # x: [B, T, n_mels]  (transposed in QuasTransformer.forward)
        x = x.permute(0, 2, 1)           # [B, n_mels, T]
        x = F.gelu(self.input_proj(x))    # [B, d, T]
        x = F.gelu(self.input_proj2(x))   # [B, d, T]
        x = x.permute(0, 2, 1)           # [B, T, d]

        x = self.pos_encoding(x)          # inject position before normalising
        x = self.pre_ln(x)               # anchor residual stream

        for block in self.attn_blocks:
            x = block(x)
        x = self.post_ln(x)              # xa will arrive at decoder normalised
        return x


# ── Decoder ──────────────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """Auto-regressive decoder with cross-attention to the encoder.

    Normalization flow
    ------------------
    tok_embedding (scaled init)
         → N × ResidualAttentionBlock (self-attn with RoPE, cross-attn, FFN)
         → ln
         → tied output projection

    RoPE is applied inside self-attention Q/K only — never to the residual
    stream — so pre-norm LayerNorm in each block cannot erase it.
    """

    def __init__(self, config):
        super().__init__()
        self.tok_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # Scale down embedding init so initial activation norms are O(1) rather
        # than O(sqrt(d_model)).  Consistent with what LayerNorm expects.
        nn.init.normal_(self.tok_embedding.weight, mean=0.0, std=config.embedding_dim ** -0.5)

        # Decoder blocks: self-attention uses RoPE, cross-attention does not
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(config, cross_attn=True, use_rope=True)
             for _ in range(config.num_layers)]
        )
        self.ln = nn.LayerNorm(config.embedding_dim)

        mask = torch.empty(config.block_size, config.block_size).fill_(float("-inf")).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, xa):
        T = x.size(1)
        x = self.tok_embedding(x)        # [B, T, d]  — no separate PE call;
                                         # RoPE is applied inside self-attn Q/K

        for block in self.blocks:
            x = block(x, xa=xa, mask=self.mask[:T, :T])
        x = self.ln(x)

        # Tied output projection — weight shared with tok_embedding
        return x @ self.tok_embedding.weight.to(x.device).T


# ── Top-level model ──────────────────────────────────────────────────────────

class QuasTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = EncoderLayer(config)
        self.decoder = DecoderLayer(config)

    def forward(self, mel, tokens):
        # mel:    [B, n_mels, T]
        # tokens: [B, T_tok]
        x = mel.transpose(1, 2)          # [B, T, n_mels]
        return self.decoder(tokens, self.encoder(x))
