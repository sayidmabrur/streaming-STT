import os

import pandas as pd
import sentencepiece as spm


def train_tokenizer(tsv_path, model_prefix="commonvoice_v25_sp", vocab_size=1000):
    print(f"Training SentencePiece model with vocab size {vocab_size}...")

    # Read the TSV and extract the text data
    try:
        df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read TSV file at {tsv_path}: {e}")

    # Drop NaNs, convert to string, and lowercase
    sentences = df["sentence"].dropna().astype(str).str.lower().tolist()

    temp_txt = "spm_train_temp.txt"
    with open(temp_txt, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

    # Train SentencePiece
    # We set pad_id=0 to act as our CTC Blank Token
    # bos_id and eos_id are disabled (-1) since CTC doesn't use them
    spm.SentencePieceTrainer.train(
        input=temp_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        pad_id=0,  # Used as CTC Blank <pad>
        unk_id=1,  # <unk> token
        bos_id=-1,  # No beginning of sentence token
        eos_id=-1,  # No end of sentence token
    )

    # Clean up the temporary text file
    if os.path.exists(temp_txt):
        os.remove(temp_txt)

    print("SentencePiece training complete!")


if __name__ == "__main__":
    from model.tokenizer import Tokenizer
    from train.config import train_cfg

    # Explicitly pre-train the tokenizer when running this script
    train_tokenizer(
        tsv_path=train_cfg.tsv_path,
        model_prefix=train_cfg.tokenizer_prefix,
        vocab_size=train_cfg.vocab_size,
    )

    # Quick test to verify it works
    tokenizer = Tokenizer(tsv_path=train_cfg.tsv_path, vocab_size=train_cfg.vocab_size)
    sample_text = "hello world this is a test"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {sample_text}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    print(f"Vocab Size: {tokenizer.get_vocab_size}")
