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
    vocab_size: int = 1000
    seed: int = 42
    wandb_log: bool = False


@dataclass
class Config:
    vocab_size: int = 1000
    # hidden_size: int = 300
    embedding_dim: int = 512
    num_layers: int = 3
    batch_size: int = 16
    max_seq_len: int = 2048


# Instantiate both configurations to be imported by other modules
model_cfg = Config()
train_cfg = TrainingConfig()
