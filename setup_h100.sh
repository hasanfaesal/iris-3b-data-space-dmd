#!/usr/bin/env bash
# One-time node setup (idempotent). Run from the project root on the H100 node, which must
# already hold this repo under /home/dev (only /home/dev persists across stop_node).
#   bash setup_h100.sh && python scripts/run_verification.py --out runs/verify
# Remember: stop_node the moment verification finishes.
set -euo pipefail

IRIS_COMMIT=13fcf6a4f7fcb36561eac3daa5ad30d0b0a1822e   # same commit NOTES.md was read from
MODELS=/home/dev/models

[ -d .venv ] || python3 -m venv --system-site-packages .venv  # sees the image's preinstalled torch
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip

python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || pip install -q torch torchvision
# no wandb/dion: only needed by Iris's own training scripts
pip install -q transformers accelerate omegaconf safetensors "huggingface_hub[cli]" pillow numpy pytest

if [ ! -d vendor/iris-3b ]; then
  mkdir -p vendor
  git clone -q https://github.com/speridlabs/iris-3b.git vendor/iris-3b
fi
git -C vendor/iris-3b checkout -q "$IRIS_COMMIT"
pip install -q -e vendor/iris-3b --no-deps

# base weights (~12 GB, fp32) and the text encoder (~8 GB), both cached on /home/dev
mkdir -p "$MODELS"
export HF_HOME=/home/dev/.cache/huggingface
hf download speridlabs/iris-3b --local-dir "$MODELS/iris-3b" --exclude "depth/*" "upscaler/*" "turbo/*"
hf download Qwen/Qwen3-VL-4B-Instruct

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
import iris3b
EOF
python -m pytest -q tests
echo "setup ok; HF_HOME=$HF_HOME (export it in the shell that runs run_verification.py)"
