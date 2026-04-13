from dataclasses import dataclass

from model import Config as ModelConfig


@dataclass
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 1e-5
    tsv_path: str = "/home/henry/Projects/ASR/dataset/en/train.tsv"
    tokenizer_prefix: str = "commonvoice_v25_sp"
    vocab_size: int = 1000
    seed: int = 42


# Instantiate both configurations to be imported by other modules
model_cfg = ModelConfig()
train_cfg = TrainingConfig(tokenizer_prefix="commonvoice_v25_sp")
