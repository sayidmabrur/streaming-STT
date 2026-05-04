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
    wandb.setup(settings=wandb.Settings(mode="online"))
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
    # ── Overall charts (x-axis = epoch) ──────────────────────────────────
    wandb.define_metric("epoch")
    wandb.define_metric("overall/loss", step_metric="epoch")
    wandb.define_metric("overall/wer",  step_metric="epoch")


step = 1
for epoch in range(epochs):
    step_in_epoch = 0
    avg_wer = np.nan
    epoch_losses: list[float] = []
    epoch_wers:   list[float] = []
    pred_texts, target_texts = [], []

    # ── Per-epoch charts (x-axis = train_step within epoch) ──────────────
    if train_cfg.wandb_log:
        wandb.define_metric(f"epoch_{epoch}/train_step")
        wandb.define_metric(f"epoch_{epoch}/loss", step_metric=f"epoch_{epoch}/train_step")
        wandb.define_metric(f"epoch_{epoch}/wer",  step_metric=f"epoch_{epoch}/train_step")

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for idx, (x, y, input_lengths, target_lengths) in enumerate(pbar):
        x, y = x.to(device), y.to(device).long()
        optimizer.zero_grad()

        output = model(x)
        output_log_softmax = nn.functional.log_softmax(output, dim=-1)
        output_log_softmax = output_log_softmax.transpose(0, 1)

        loss = loss_fn(output_log_softmax, y, input_lengths, target_lengths)
        epoch_losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ── Log per-step loss for this epoch's chart ──────────────────────
        if train_cfg.wandb_log:
            wandb.log({
                f"epoch_{epoch}/loss":       loss.item(),
                f"epoch_{epoch}/train_step": step_in_epoch,
            })

        if step % 10000 == 0:
            model.eval()
            wers = []
            test_pbar = tqdm(test_dataloader, desc="Testing")
            for _, (tx, ty, t_input_lengths, _t_target_lengths) in enumerate(test_pbar):
                tx, ty = tx.to(device), ty.to(device).long()
                with torch.no_grad():
                    t_output = model(tx)
                    pred_texts, _ = greedy_decode(t_output, tokenizer, t_input_lengths)
                target_texts = target_decode(ty, tokenizer)
                wer = compute_wer(pred_texts, target_texts)
                wers.append(wer)
            avg_wer = sum(wers) / len(wers) if len(wers) > 0 else np.nan
            epoch_wers.append(avg_wer)
            print(f"avg_wer: {avg_wer:.4f}")

            # ── Log per-step WER for this epoch's chart ───────────────────
            if train_cfg.wandb_log:
                wandb.log({
                    f"epoch_{epoch}/wer":        avg_wer,
                    f"epoch_{epoch}/train_step": step_in_epoch,
                })

            model.train()

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        step += 1
        step_in_epoch += 1

    # ── End-of-epoch: log to overall charts ──────────────────────────────
    epoch_avg_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
    epoch_avg_wer  = float(np.mean(epoch_wers))   if epoch_wers   else np.nan
    if train_cfg.wandb_log:
        wandb.log({
            "overall/loss": epoch_avg_loss,
            "overall/wer":  epoch_avg_wer,
            "epoch":        epoch,
        })

    save_file(model.state_dict(), f"checkpoint_{epoch}.safetensors")

    print(f"\n=== Epoch {epoch} Summary ===")
    print(f"Loss: {loss.item():.4f} | Avg WER: {avg_wer:.4f}")
    print(f"target_texts: {target_texts[-5:]}")
    print(f"pred_texts: {pred_texts[-5:]}")
    print("=============================\n")

if train_cfg.wandb_log:
    run.finish()
