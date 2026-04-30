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

import wandb
from model import Config, QuasTransformer
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

model = QuasTransformer(model_cfg).to(device)
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


def greedy_decode(logits, tokenizer, input_lengths=None):
    # logits shape: (batch, time, vocab)
    preds = torch.argmax(logits, dim=-1)  # (batch, time)
    decoded = []
    total_frames = 0
    blank_frames = 0
    for i in range(preds.shape[0]):
        # Trim to valid (non-padded) frames only
        valid_len = int(input_lengths[i]) if input_lengths is not None else preds.shape[1]
        pred = preds[i, :valid_len].tolist()

        total_frames += len(pred)
        blank_frames += pred.count(0)

        # Standard CTC collapse: merge consecutive duplicates, then remove blanks
        pred_seq = []
        for j in range(len(pred)):
            if pred[j] != 0 and (j == 0 or pred[j] != pred[j - 1]):
                pred_seq.append(pred[j])

        text = tokenizer.decode(pred_seq)
        decoded.append(text if text.strip() else "<empty>")

    blank_ratio = blank_frames / total_frames if total_frames > 0 else 1.0
    return decoded, blank_ratio


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
    num_workers=2,       # 4 workers × prefetch × pinned copies was the RAM spike
    pin_memory=True,
    prefetch_factor=2,   # only 2 batches staged per worker at a time
    persistent_workers=True,  # avoids re-spawning workers each epoch
)

test_dataset = CommonVoiceDataset(
    tsv_path=train_cfg.test_tsv_path,
    audio_dir_path=train_cfg.audio_dir_path,
    transform=FeatureExtractor(),
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=model_cfg.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

# Run in offline mode to avoid connection drops during training.
# After training finishes, sync to W&B with:
#   wandb sync wandb/latest-run

if train_cfg.wandb_log:
    wandb.setup(settings=wandb.Settings(mode="offline"))
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="sayid-10121012-universitas-komputer-indonesia",
        # Set the wandb project where this run will be logged.
        project="QuasASR",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": train_cfg.learning_rate,
            "architecture": "Transformer_Encoder",
            "dataset": "CommonVoice-v2.5",
            "epochs": train_cfg.epochs,
        },
    )
else:
    glob_log = []

    def run(log: list, step: int):
        glob_log.append({"step": step, "log": log})


step = 1
for epoch in range(epochs):
    avg_wer = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for idx, (x, y, input_lengths, target_lengths) in enumerate(pbar):
        x, y = x.to(device), y.to(device).long()
        optimizer.zero_grad()

        output = model(x)
        output_log_softmax = nn.functional.log_softmax(output, dim=-1)
        output_log_softmax = output_log_softmax.transpose(0, 1)

        loss = loss_fn(output_log_softmax, y, input_lengths, target_lengths)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10000 == 0:
            model.eval()
            wers = []
            pred_texts, target_texts = [], []
            test_pbar = tqdm(test_dataloader, desc="Testing")
            blank_ratios = []
            for idx, (x, y, input_lengths, target_lengths) in enumerate(test_pbar):
                x, y = x.to(device), y.to(device).long()
                with torch.no_grad():
                    output = model(x)
                    pred_texts, blank_ratio = greedy_decode(output, tokenizer, input_lengths)
                target_texts = target_decode(y, tokenizer)
                wer = compute_wer(pred_texts, target_texts)
                wers.append(wer)
                blank_ratios.append(blank_ratio)
            avg_wer = sum(wers) / len(wers) if len(wers) > 0 else np.nan
            avg_blank = sum(blank_ratios) / len(blank_ratios) if blank_ratios else 1.0
            print(f"avg_wer: {avg_wer:.4f} | blank_ratio: {avg_blank:.3f}")
            # if avg_blank > 0.99:
            #     print("  [!] Model is collapsing to blank — still in early CTC training phase.")
            # for pred, target in zip(pred_texts[-5:], target_texts[-5:]):
            #     print(f"pred: {pred}")
            #     print(f"target: {target}")
            #     print("-"*5)

            model.train()
        #     run.log({"wer": avg_wer}, step=step)
        # run.log({"loss": loss.item()}, step=step)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        step += 1

    save_file(model.state_dict(), f"checkpoint_{epoch}.safetensors")

    print(f"\n=== Epoch {epoch} Summary ===")
    print(f"Loss: {loss.item():.4f} | Avg, WER: {avg_wer:.4f}")
    print(f"target_texts: {target_texts[-5:]}")
    print(f"pred_texts: {pred_texts[-5:]}")
    print("=============================\n")



# run.finish()
