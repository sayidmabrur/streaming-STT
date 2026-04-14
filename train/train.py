import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.functional as F_audio
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ASRModel
from model.tokenizer import Tokenizer

from .config import model_cfg, train_cfg
from .dataset import CommonVoiceDataset, FeatureExtractor, collate_fn

# Set random seeds for reproducibility
torch.manual_seed(train_cfg.seed)
random.seed(train_cfg.seed)
np.random.seed(train_cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(train_cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = train_cfg.epochs

model = ASRModel(model_cfg).to(device)
optimizer = optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

dataset = CommonVoiceDataset(
    tsv_path=train_cfg.tsv_path,
    audio_dir_path=train_cfg.audio_dir_path,
    transform=FeatureExtractor(),
)
tokenizer = Tokenizer(
    tsv_path=train_cfg.tsv_path,
    vocab_size=train_cfg.vocab_size,
    model_prefix=train_cfg.tokenizer_prefix,
)


def greedy_decode(logits, tokenizer):
    # logits shape: (batch, time, vocab)
    preds = torch.argmax(logits, dim=-1)
    decoded = []
    for i in range(preds.shape[0]):
        pred = preds[i].tolist()
        pred_seq = []
        for j in range(len(pred)):
            if pred[j] != 0 and (j == 0 or pred[j] != pred[j - 1]):
                pred_seq.append(pred[j])

        text = tokenizer.decode(pred_seq)
        decoded.append(text if text.strip() else "<empty>")
    return decoded


def target_decode(targets, tokenizer):
    decoded = []
    for i in range(targets.shape[0]):
        tgt_seq = [t.item() for t in targets[i] if t.item() != 0]
        decoded.append(tokenizer.decode(tgt_seq))
    return decoded


def compute_wer(preds, targets):
    wer = 0.0
    valid_samples = 0
    for p, t in zip(preds, targets):
        p = re.sub(r"[^\w\s]", "", p.lower())
        t = re.sub(r"[^\w\s]", "", t.lower())
        p_words = p.split()
        t_words = t.split()
        if len(t_words) == 0:
            continue
        dist = F_audio.edit_distance(p_words, t_words)
        wer += dist / len(t_words)
        valid_samples += 1
    return wer / valid_samples if valid_samples > 0 else 0.0


dataloader = DataLoader(
    dataset,
    batch_size=model_cfg.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

for epoch in range(epochs):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for idx, (x, y, input_lengths, target_lengths) in enumerate(pbar):
        x, y = x.to(device), y.to(device).long()
        optimizer.zero_grad()

        output = model(x)
        output_log_softmax = nn.functional.log_softmax(output, dim=-1)
        output_log_softmax = output_log_softmax.transpose(
            0, 1
        )  # (time, batch, vocab_size)

        loss = loss_fn(output_log_softmax, y, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # save the model epoch checkpoint
    #
    save_file(model.state_dict(), f"checkpoint_{epoch}.safetensors")
    model.eval()
    with torch.no_grad():
        output = model(x)
        pred_texts = greedy_decode(output, tokenizer)
        target_texts = target_decode(y, tokenizer)
        wer = compute_wer(pred_texts, target_texts)
    model.train()

    print(f"\n=== Epoch {epoch} Summary ===")
    print(f"Loss: {loss.item():.4f} | WER: {wer:.4f}")
    if len(pred_texts) > 0:
        print(f"  Target: {target_texts[0]}")
        print(f"  Pred:   {pred_texts[0]}")
    print("=============================\n")
