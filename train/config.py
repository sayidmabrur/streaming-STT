from dataclasses import dataclass

from model import Config as ModelConfig


@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-5
    tsv_path: str = "/home/capon/projects/epic-projects/streaming-STT/dataset/cv-corpus-25.0-2026-03-09/en/train.tsv"
    audio_dir_path: str = "/home/capon/projects/epic-projects/streaming-STT/dataset/cv-corpus-25.0-2026-03-09/en/clips"
    tokenizer_prefix: str = "commonvoiceBPE"
    vocab_size: int = 1000
    seed: int = 42


# Instantiate both configurations to be imported by other modules
model_cfg = ModelConfig()
train_cfg = TrainingConfig()
