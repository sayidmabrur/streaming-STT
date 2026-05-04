import os

import sentencepiece as spm


class Tokenizer:
    def __init__(
        self,
        tsv_path="/home/capon/projects/epic-projects/ASR/dataset/en/train.tsv",
        vocab_size=1000,
        model_prefix="tokenizer",
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

    # ── special token ids ────────────────────────────────────────────────
    @property
    def bos_id(self) -> int:
        """<s> token id.  Falls back to 0 (PAD) when the SP model was trained
        without BOS (bos_id=-1), so we never feed -1 into an embedding table."""
        bid = self.sp.bos_id()
        return bid if bid >= 0 else 0

    @property
    def eos_id(self) -> int:
        """</s> token id.  Falls back to 0 (PAD) when the SP model was trained
        without EOS (eos_id=-1)."""
        eid = self.sp.eos_id()
        return eid if eid >= 0 else 0

    @property
    def pad_id(self) -> int:
        return 0  # collate_fn pads with 0; train/tokenizer.py sets pad_id=0

    # ── encode / decode ──────────────────────────────────────────────────
    @property
    def has_special_tokens(self) -> bool:
        """True only when BOS and EOS are real, distinct tokens (i.e. the
        tokenizer was retrained with bos_id=2 / eos_id=3).  When the old
        CTC tokenizer is loaded, bos_id == eos_id == pad_id == 0, so
        prepending/appending them would corrupt every sequence."""
        return self.bos_id != self.pad_id and self.eos_id != self.pad_id

    def encode(self, text: str) -> list[int]:
        """Returns [BOS, tok1, …, tokN, EOS] when the tokenizer has real
        special tokens; otherwise returns bare [tok1, …, tokN]."""
        ids = self.sp.encode_as_ids(text.lower())
        if self.has_special_tokens:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        # Strip special tokens before decoding
        filtered = [t for t in ids if t not in (self.bos_id, self.eos_id, self.pad_id)]
        return self.sp.decode_ids(filtered)

    @property
    def get_vocab_size(self):
        return self.sp.vocab_size()
