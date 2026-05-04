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
# ignore_index=0 skips PAD positions (collate_fn pads tokens with 0)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

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


def greedy_decode_autoregressive(model, mel, tokenizer, max_len: int = 100):
    """Autoregressive greedy decode for the encoder-decoder model.

    The encoder is run ONCE and its output is reused across all token steps.
    Previously model(mel, tokens) was called inside the loop, recomputing
    the full encoder up to max_len times per batch.
    """
    has_special = tokenizer.has_special_tokens
    bos_id      = tokenizer.bos_id
    eos_id      = tokenizer.eos_id
    pad_id      = tokenizer.pad_id
    B           = mel.size(0)

    # ── Encode audio ONCE ────────────────────────────────────────────────
    with torch.no_grad():
        xa = model.encoder(mel.transpose(1, 2))   # [B, T_enc, d]

    tokens   = torch.full((B, 1), bos_id, dtype=torch.long, device=mel.device)
    finished = torch.zeros(B, dtype=torch.bool,  device=mel.device)
    stop_tok = eos_id if has_special else pad_id

    for _ in range(max_len):
        with torch.no_grad():
            logits   = model.decoder(tokens, xa)                          # [B, T, V]
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)      # [B, 1]
        tokens   = torch.cat([tokens, next_tok], dim=1)
        finished |= next_tok.squeeze(1) == stop_tok
        if finished.all():
            break

    decoded = []
    for i in range(B):
        seq = tokens[i, 1:].tolist()        # drop seed BOS token
        if stop_tok in seq:
            seq = seq[:seq.index(stop_tok)]
        text = tokenizer.decode(seq)
        decoded.append(text if text.strip() else "<empty>")
    return decoded


def target_decode(targets, tokenizer):
    """Decode padded target token tensors, correctly handling both tokenizer
    variants.

    Old tokenizer (no BOS/EOS): sequences are [tok1, …, tokN, 0, 0, …]
        → collect until first 0.
    New tokenizer (BOS=2, EOS=3): sequences are [2, tok1, …, tokN, 3, 0, 0, …]
        → skip leading BOS, collect until EOS or PAD.
    """
    has_special = tokenizer.has_special_tokens
    bos_id      = tokenizer.bos_id
    eos_id      = tokenizer.eos_id
    decoded     = []
    for i in range(targets.shape[0]):
        seq    = []
        tokens = targets[i].tolist()
        start  = 1 if has_special and tokens and tokens[0] == bos_id else 0
        for tid in tokens[start:]:
            if tid == 0 or (has_special and tid == eos_id):
                break   # hit PAD (always) or EOS (new tokenizer)
            seq.append(tid)
        decoded.append(tokenizer.decode(seq))
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

        # Teacher-forcing: decoder sees [BOS, t1, …, tN-1], predicts [t1, …, tN, EOS]
        decoder_input = y[:, :-1]               # [B, T-1]
        targets       = y[:, 1:].contiguous()   # [B, T-1]

        output = model(x, decoder_input)        # [B, T-1, vocab_size]
        B_sz, T_sz, V_sz = output.shape
        loss = loss_fn(output.reshape(B_sz * T_sz, V_sz), targets.reshape(B_sz * T_sz))
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
            for _, (tx, ty, _t_input_lengths, _t_target_lengths) in enumerate(test_pbar):
                tx = tx.to(device)
                ty = ty.to(device).long()
                pred_texts   = greedy_decode_autoregressive(model, tx, tokenizer)
                target_texts = target_decode(ty, tokenizer)
                wer = compute_wer(pred_texts, target_texts)
                wers.append(wer)
            avg_wer = sum(wers) / len(wers) if len(wers) > 0 else np.nan
            epoch_wers.append(avg_wer)
            print(f"avg_wer: {avg_wer:.4f}")
            print(f"pred: {pred_texts[0]}, target: {target_texts[0]}")

            # ── Log per-step WER for this epoch's chart ───────────────────
            if train_cfg.wandb_log:
                wandb.log({
                    f"epoch_{epoch}/wer":        avg_wer,
                    f"epoch_{epoch}/train_step": step_in_epoch,
                })

            model.train()
            torch.cuda.empty_cache()   # release eval tensors before resuming training

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
