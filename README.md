# Streaming ASR 🎙️

This project explores lightweight, efficient methods for online Automatic Speech Recognition (ASR) using Open Speech Data. The goal is to build a fast and robust streaming ASR system capable of real-time or near real-time inference.

## 📊 Dataset

We use the [Mozilla Common Voice](https://commonvoice.mozilla.org/) dataset:
* **Version:** Commonvoice-v25-en (English)
* **Features:** 128-bin Log-Mel Spectrograms (16kHz resampling)
* **Tokenization:** SentencePiece (Unigram/BPE)

## 🧠 Model Architecture

The current acoustic model is designed to be lightweight and fast:
* **Encoder:** 3-layer Stacked Bidirectional LSTM
* **Feature Projection:** Linear layer mapping 128 Mel bins to the hidden embedding dimension.
* **Loss Function:** CTC (Connectionist Temporal Classification)
* **Decoding:** CTC Greedy Search

## 🚀 Getting Started

### Prerequisites
Ensure your environment has the required dependencies (PyTorch, Torchaudio, SentencePiece, etc.). If using Conda, activate your environment:
```bash
conda activate llm-env
```

### Training
To train the model from scratch, run the training script as a module:
```bash
python -m train.train
```

### Inference
To run inference on a sample audio file and measure latency:
```bash
python inference/infer.py
```

---

## 📈 Results & Performance

### Training Progression (3 Epochs)
A quick "blind train" run over 3 epochs demonstrates the model rapidly learning to align acoustic features with character tokens:

**Epoch 1:**
* **Loss:** `4.7893` | **WER:** `0.8500`
* **Target:** `stolaroff served on the board of directors of the albert hofmann foundation.`
* **Pred:** `some of the of the asation.`

**Epoch 2:**
* **Loss:** `3.8338` | **WER:** `0.7167`
* **Target:** `noseworthy was an active parliamentarian and defended the rights of immigrants and minorities.`
* **Pred:** `the was byment time of the for the in trans news.`

**Epoch 3:**
* **Loss:** `3.5094` | **WER:** `0.7509`
* **Target:** `his earliest works were in english and concerned with the glorious revolution.`
* **Pred:** `is the a periodian americanster and pre.`

### Inference Latency
The stacked LSTM architecture combined with greedy decoding yields highly efficient inference times:

```text
========================================
Inference Result:
========================================
['somely other the center home was secondly the bo fo pa.']
----------------------------------------
Latency: 0.2007 seconds
========================================
```

## 🗺️ Future Work / Roadmap

- [ ] **Beam Search Decoding:** Upgrade from greedy search to CTC Beam Search to improve Word Error Rate (WER) without retraining.
- [ ] **Language Model Fusion:** Integrate an N-gram or small LLM during beam search to correct spelling and grammar mistakes dynamically.
- [ ] **Streaming Inference Optimization:** Implement stateful LSTMs / chunk-based processing for true real-time streaming constraints.