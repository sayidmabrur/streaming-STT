import os

import sentencepiece as spm


class Tokenizer:
    def __init__(
        self,
        tsv_path="/home/henry/Projects/ASR/dataset/en/train.tsv",
        vocab_size=1000,
        model_prefix="commonvoice_v25_sp",
    ):
        self.tsv_path = tsv_path
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.model_file = f"{self.model_prefix}.model"

        # Model must be pre-trained explicitly!
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(
                f"Tokenizer model '{self.model_file}' not found. "
                "Please pre-train the tokenizer by running: python -m train.tokenizer"
            )

        # Load the trained SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_file)

    def encode(self, text: str) -> list[int]:
        # Lowercase text to match training data
        return self.sp.encode_as_ids(text.lower())

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode_ids(ids)

    @property
    def get_vocab_size(self):
        return self.sp.vocab_size()
