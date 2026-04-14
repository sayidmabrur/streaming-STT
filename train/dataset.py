import os

import librosa
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from model.tokenizer import Tokenizer
from train.config import train_cfg


class CommonVoiceDataset(Dataset):
    def __init__(
        self, tsv_path, audio_dir_path=None, transform=None, target_transform=None
    ):
        if audio_dir_path is None:
            self.audio_dir_path = (
                "/home/capon/projects/epic-projects/ASR/dataset/en/clips/"
            )
        else:
            self.audio_dir_path = audio_dir_path

        self.tsv_path = tsv_path
        self.transform = transform
        self.target_transform = target_transform

        # Load our new SentencePiece Tokenizer
        self.tokenizer = Tokenizer(
            tsv_path=self.tsv_path, model_prefix=train_cfg.tokenizer_prefix
        )

        self.df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
        self.df = self.df.dropna(subset=["path", "sentence"]).reset_index(drop=True)

        # Typically use float32 for transforms, can cast to bfloat16 later if needed for the model
        self.dtype = torch.float32
        print(f"Loaded {len(self.df)} records from TSV")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio_path = os.path.join(self.audio_dir_path, row["path"])
        y, sr = torchaudio.load(audio_path)

        # torchaudio returns shape [channels, time]. Squeeze channel dim to match librosa's 1D output
        if y.shape[0] == 1:
            y = y.squeeze(0)

        y = y.to(self.dtype)

        sentence = str(row["sentence"])
        tokenized_sentence = self.tokenizer.encode(sentence)
        token = torch.tensor(tokenized_sentence, dtype=torch.long)

        if self.transform:
            y = self.transform(y)

        return (y, token)


class FeatureExtractor(nn.Module):
    def __init__(self, input_freq=48000, resample_freq=16000, n_fft=1024, n_mel=128):
        super(FeatureExtractor, self).__init__()

        # Resample to 16kHz (standard for ASR)
        self.resample = T.Resample(orig_freq=input_freq, new_freq=resample_freq)

        # Extract MelSpectrogram directly
        self.mel_spec = T.MelSpectrogram(
            sample_rate=resample_freq, n_fft=n_fft, n_mels=n_mel
        )

        # Convert to Log scale (crucial for ASR)
        self.amplitude_to_DB = T.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, x):
        x = self.resample(x)
        x = self.mel_spec(x)
        x = self.amplitude_to_DB(x)

        # Instance Normalization: Standardize mean to 0 and variance to 1 per-spectrogram
        x = (x - x.mean()) / (x.std() + 1e-7)

        return x


def collate_fn(batch):

    specs = []
    tokens = []
    input_lengths = []
    target_lengths = []

    for spec, token in batch:
        spec = spec.squeeze(0)

        spec = spec.transpose(0, 1)

        specs.append(spec)
        tokens.append(token)
        input_lengths.append(spec.shape[0])
        target_lengths.append(token.shape[0])

    specs = pad_sequence(specs, batch_first=True, padding_value=0)

    specs = specs.transpose(1, 2)

    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)

    return (
        specs,
        tokens,
        torch.tensor(input_lengths, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
    )
