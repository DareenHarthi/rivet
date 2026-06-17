# RIVET

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/submit/7726288/view)
[![Demo](https://img.shields.io/badge/🔊%20Demo-RIVET-blue)](https://username.github.io/rivet)

RIVET is a robust voice attribute editor that edits speaker **age** and **gender** while preserving speaker identity.
It introduces two key innovations: an *idempotency* constraint (`f(f(x)) = f(x)`) that acts as an implicit regularizer, and a conditional normalizing flow over speaker embeddings that makes editing robust to the noisy, inconsistent attribute labels found in large-scale speech corpora.

**Paper:** Robust Idempotent Voice Attribute Editing ([arXiv](https://arxiv.org/submit/7726288/view)) — Interspeech 2026.  
**Authors:** Dareen Alharthi, Bhuvan Koduru, Rita Singh, Bhiksha Raj — Carnegie Mellon University

<!-- <img width="100%" alt="rivet_overview" src="..."> -->

## How It Works

RIVET jointly trains three components:

1. **ECAPA-TDNN** — extracts a fixed-length speaker embedding and predicts age/gender with noisy-label-robust classification heads.
2. **SpeakerFlow** — a conditional normalizing flow that maps the speaker embedding into an age- and gender-conditioned latent space, so attributes can be edited by re-sampling under target conditions.
3. **VITS generator** — reconstructs speech conditioned on the (edited) speaker embedding.

During training, synthesized speech is re-encoded and re-synthesized; an MSE loss between the original and re-synthesized representations (both the latent `z` and the speaker embedding) enforces idempotency, stabilizing edits under noisy supervision.

## Quick Start

Edit the age and gender of samples drawn from a filelist using a trained checkpoint:

```bash
python edit.py \
  --config configs/globe_base.json \
  --checkpoint_dir logs/globe_base \
  --checkpoint_step 71000 \
  --data_file filelists/test.txt \
  --output_dir edited_samples \
  --num_samples 20 \
  --device cuda
```

For each sample this writes age- and gender-edited WAVs to `output_dir/`, along with a `.meta.txt` describing each transformation and a self-contained `demo.html` for quick listening.

Output naming:
```
<basename>_age<src>to<tgt>.wav   # age edit
<basename>_sex<S>to<T>.wav       # gender edit (M / F)
```

## Setup

The environment matches [VITS](https://github.com/jaywalnut310/vits). Requires Python 3.8+ and a CUDA-capable GPU.

```bash
pip install -r requirements.txt

# Text frontend (phonemizer backend)
sudo apt-get install espeak-ng

# Monotonic alignment search (compile once)
cd monotonic_align && python setup.py build_ext --inplace && cd ..
```

## Data Preparation

Each filelist line has five `|`-separated fields:

```
/path/to/audio.wav|speaker_id|AGE|SEX|transcript text
```

- **speaker_id** — integer speaker index
- **AGE** — integer in years (binned by decade internally, e.g. 20s, 30s, …)
- **SEX** — `0` = male, `1` = female

Audio is expected at the `sampling_rate` set in the config (22.05 kHz by default).

Clean and phonemize the transcripts before training (writes a `.cleaned` copy next to each filelist):

```bash
python preprocess.py \
  --filelists filelists/train.txt filelists/val.txt \
  --text_index 4 \
  --text_cleaners english_cleaners2
```

Then point `data.training_files` / `data.validation_files` in the config to the `.cleaned` filelists.

Datasets used in the paper:

| Dataset | Description | Notes |
|---------|-------------|-------|
| [GLOBE](https://arxiv.org/abs/2406.14875) | ~535 h, 23,519 speakers, 164 global accents | Main training set |
| [EARS](https://arxiv.org/abs/2406.04923) | ~7 h, high-quality speech | Label-noise robustness experiments |

## Train RIVET

`train_rivet.py` reads the config (`-c`) and a model/run name (`-m`), and saves checkpoints to `logs/<model_name>/`. It launches one process per visible GPU automatically — no `torchrun` needed.

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python train_rivet.py -c configs/globe_base.json -m globe_base

# Multiple GPUs (uses all visible devices)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_rivet.py -c configs/globe_base.json -m globe_base
```

Checkpoints are written every `train.eval_interval` steps and training resumes automatically from the latest checkpoints in `logs/<model_name>/`:

```
G_<step>.pth        # VITS generator
D_<step>.pth        # discriminator
ECAPA_<step>.pth    # speaker encoder
CNF_<step>.pth      # SpeakerFlow (normalizing flow)
NOISE_<step>.pth    # noise-matrix optimizer state
```

Configs:

| Config | Dataset | Notes |
|--------|---------|-------|
| `configs/globe_base.json` | GLOBE | Main config (256-dim model) |
| `configs/ears.json` | EARS | Smaller model (192-dim), label-noise experiments |

## Repository Structure

```
rivet/
├── train_rivet.py     # Training script (DDP, multi-GPU)
├── edit.py            # Editing / inference
├── preprocess.py      # Text cleaning and phonemization
├── models.py          # SynthesizerTrn, SpeakerFlow, MultiPeriodDiscriminator
├── ecapa.py           # ECAPA-TDNN speaker encoder + noisy-label classifiers
├── modules.py         # Flow layers and convolutional building blocks
├── attentions.py      # Multi-head attention and encoder/decoder
├── losses.py          # Generator, discriminator, feature, and KL losses
├── transforms.py      # Piecewise rational quadratic spline transforms
├── mel_processing.py  # STFT and mel-spectrogram utilities
├── commons.py         # Init, masking, and padding helpers
├── utils.py           # Checkpoints, logging, hyperparameters
├── data_utils.py      # Dataset loader, SpecAugment, bucket sampler
├── text/              # Text cleaners and symbol vocabulary
├── monotonic_align/   # Cython monotonic alignment (compile before use)
├── configs/           # globe_base.json, ears.json
└── filelists/         # Training / validation filelists
```

## Acknowledgements

This code builds on [VITS](https://github.com/jaywalnut310/vits) and an [ECAPA-TDNN](https://github.com/speechbrain/speechbrain) implementation.

## Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{alharthi2026rivet,
  title     = {{RIVET}: Robust Idempotent Voice Attribute Editing},
  author    = {Alharthi, Dareen and Koduru, Bhuvan and Singh, Rita and Raj, Bhiksha},
  booktitle = {Proc. Interspeech 2026},
  year      = {2026},
}
```
