#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib scikit-learn

# Official Mamba is optional. If this fails, train.py falls back to LiteMamba.
pip install "mamba-ssm[causal-conv1d]" --no-build-isolation || true

python - <<'PY'
import importlib
import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
for name in ["mamba_ssm", "causal_conv1d"]:
    try:
        mod = importlib.import_module(name)
        print(name, "OK", getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print(name, "UNAVAILABLE", type(exc).__name__, exc)
PY
