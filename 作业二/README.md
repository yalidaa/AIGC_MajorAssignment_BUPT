# Assignment 2: Vision Mamba on Fashion-MNIST Binary Classification

This assignment continues to use the previous Fashion-MNIST binary dataset:

- Class 0: Sneaker, original Fashion-MNIST label 7
- Class 1: Bag, original Fashion-MNIST label 8

The goal is to build a Mamba-based image classifier and compare it with a
Transformer classifier on the same dataset.

## Structure

```text
作业二/
  dataset.py
  models.py
  train.py
  compare_results.py
  requirements.txt
  setup_wsl.sh
```

## Model Design

Images are converted into token sequences by patch embedding:

```text
28x28 image -> 4x4 patches -> 49 patch tokens -> class token -> classifier
```

`VisionMambaClassifier` first tries to use the official `mamba-ssm` backend.
If `mamba-ssm` is unavailable, it uses `LiteMambaBlock`, a lightweight
Mamba-style fallback block with depthwise 1D sequence convolution, gating, and
state-conditioned selective modulation.

`VisionTransformerClassifier` uses standard multi-head self-attention blocks
with the same patch embedding and classification head style.

## WSL Setup

If WSL does not have `pip` or `venv`, install them first:

```bash
sudo apt update
sudo apt install python3-pip python3-venv build-essential
```

Then run:

```bash
cd "/mnt/e/基于 Fashion-MNIST 的轻量级神经网络二分类——Bag 与 Sneaker 识别/作业二"
bash setup_wsl.sh
```

Official Mamba may fail if the CUDA build toolchain is incomplete. The training
code remains usable because it automatically falls back to the lightweight
Mamba-style implementation.

## Training

Run Mamba:

```bash
source .venv/bin/activate
python train.py --model mamba --epochs 10 --batch-size 128
```

Force the lightweight fallback:

```bash
python train.py --model mamba --force-lite-mamba --epochs 10 --batch-size 128
```

Run Transformer:

```bash
python train.py --model transformer --epochs 10 --batch-size 128
```

Compare results:

```bash
python compare_results.py
```

Generated outputs are saved under `runs/`.
