from dataclasses import dataclass


@dataclass
class TrainingConfig:
    epochs: int = 12
    learning_rate: float = 1e-6
    tsv_path: str = "/home/capon/projects/epic-projects/streaming-STT/dataset/cv-corpus-25.0-2026-03-09/en/train.tsv"
    test_tsv_path: str = "/home/capon/projects/epic-projects/streaming-STT/dataset/cv-corpus-25.0-2026-03-09/en/test.tsv"
    audio_dir_path: str = "/home/capon/projects/epic-projects/streaming-STT/dataset/cv-corpus-25.0-2026-03-09/en/clips"
    tokenizer_prefix: str = "commonvoiceBPE"
    vocab_size: int = 3000
    seed: int = 42


@dataclass
class Config:
    vocab_size: int = 3000
    # hidden_size: int = 300
    embedding_dim: int = 768
    num_layers: int = 8
    batch_size: int = 16
    max_seq_len: int = 2048


# Instantiate both configurations to be imported by other modules
model_cfg = Config()
train_cfg = TrainingConfig()
