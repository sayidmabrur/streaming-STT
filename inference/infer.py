import os
import sys
import time

import torch
import torchaudio
from safetensors.torch import load_file

# Add the parent directory to the path so we can import from model and train
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import ASRModel, Config
from model.tokenizer import Tokenizer
from train.dataset import FeatureExtractor


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
        decoded.append(text)
    return decoded


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model Config and Architecture
    config = Config()
    model = ASRModel(config).to(device)

    # 2. Load the trained weights
    checkpoint_path = "checkpoint_0.safetensors"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model weights loaded successfully.")

    # 3. Setup Tokenizer and Feature Extractor
    # Note: Make sure the path to the TSV is correct for your system
    tokenizer = Tokenizer(tsv_path="../dataset/en/train.tsv")
    feature_extractor = FeatureExtractor().to(device)

    # 4. Load a sample audio file
    audio_path = "dataset/en/test_audio/common_voice_en_34.mp3"  # Change this to your actual file name
    if not os.path.exists(audio_path):
        print(
            f"Audio file {audio_path} not found. Please create the folder and add an audio file."
        )
        return

    # torchaudio loads as (channels, time)
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform[0, :].unsqueeze(0)  # Convert to mono if stereo

    waveform = waveform.to(device)

    # 5. Extract features
    start_time = time.time()
    with torch.no_grad():
        # Shape becomes (channels, n_mels, time)
        mel_spectrogram = feature_extractor(waveform)

        # Squeeze channel dim and add batch dim -> (batch, n_mels, time)
        mel_spectrogram = mel_spectrogram.squeeze(0).unsqueeze(0)

        # 6. Run Inference
        logits = model(mel_spectrogram)

        # 7. Decode
        transcription = greedy_decode(logits, tokenizer)

    end_time = time.time()
    latency = end_time - start_time

    print("\n" + "=" * 40)
    print("Inference Result:")
    print("=" * 40)
    print(transcription)
    print("-" * 40)
    print(f"Latency: {latency:.4f} seconds")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
