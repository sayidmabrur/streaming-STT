import os
from dataclasses import dataclass

# Define the project root automatically assuming config.py is in ASR/train/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TrainingConfig:
    epochs: int = 12
    learning_rate: float = 1e-6
    tsv_path: str = os.path.join(PROJECT_ROOT, "dataset/en/train.tsv")
    test_tsv_path: str = os.path.join(PROJECT_ROOT, "dataset/en/test.tsv")
    audio_dir_path: str = os.path.join(PROJECT_ROOT, "dataset/en/clips")
    tokenizer_prefix: str = "commonvoiceBPE"
    vocab_size: int = 1000    # must match the retrained tokenizer (train/tokenizer.py)
    seed: int = 42
    wandb_log: bool = True


@dataclass
class Config:
    vocab_size: int = 1000    # must match train/tokenizer.py vocab_size
    embedding_dim: int = 512
    num_layers: int = 8
    batch_size: int = 16

    # ── Decoder: max token-sequence length ───────────────────────────────
    # Causal mask is [block_size × block_size] float32 — keep this small.
    # 448 fits the longest BPE-1000 transcription and costs < 1 MB.
    block_size: int = 10240

    # ── Encoder: max mel-frame length ────────────────────────────────────
    # collate_fn caps audio at MAX_FRAMES=1024; use 1024 here so the
    # sinusoidal PE table always covers the full encoder input.
    # This is independent of block_size (which controls the decoder).
    max_audio_frames: int = 1024

    num_heads: int = 8
    dropout: float = 0.01
    n_mels: int = 128         # must match FeatureExtractor n_mel (default 128)

# Instantiate both configurations to be imported by other modules
model_cfg = Config()
train_cfg = TrainingConfig()
