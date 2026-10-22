#!/usr/bin/env bash
# One-time node setup (idempotent). Run from the project root on the H100 node, which must
# already hold this repo under /home/dev (only /home/dev persists across stop_node).
#   bash setup_h100.sh && python scripts/run_verification.py --checkpoint /home/dev/models/iris-3b --out runs/verify
# Remember: stop_node the moment verification finishes (unless caching embeddings next).
#
# Dependencies come from the HF wheels repo (hasanfaesal/iris-wheels-temp): the node's
# egress to PyPI-family hosts measured ~20 KB/s while HF measures ~30 MB/s. The
# torch/torchvision/nvidia/triton wheels are intentionally NOT installed: this image
# ships torch 2.13+cu129, and the uploaded torch 2.7.1+cu126 stack would downgrade it.
set -euo pipefail

IRIS_COMMIT=13fcf6a4f7fcb36561eac3daa5ad30d0b0a1822e   # same commit NOTES.md was read from
MODELS=/home/dev/models
WHEELS=/home/dev/wheels
WHEELS_REPO=https://huggingface.co/hasanfaesal/iris-wheels-temp/resolve/main/wheels

export HF_HOME=/home/dev/.cache/huggingface
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-180}" PIP_RETRIES="${PIP_RETRIES:-10}"

cd /home/dev/h100-2
[ -d .venv ] || python3 -m venv --system-site-packages .venv  # sees the image's preinstalled torch
# shellcheck disable=SC1091
source .venv/bin/activate

# small wheels only; torch-stack wheels deliberately excluded (see header)
WHEEL_NAMES=(
  accelerate-1.15.0-py3-none-any.whl
  annotated_doc-0.0.5-py3-none-any.whl
  anyio-4.15.1-py3-none-any.whl
  click-8.5.0-py3-none-any.whl
  filelock-4.2.4-py3-none-any.whl
  fsspec-2026.9.0-py3-none-any.whl
  h11-0.16.0-py3-none-any.whl
  hf_xet-1.7.0-cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
  httpcore2-2.16.0-py3-none-any.whl
  httpx2-2.16.0-py3-none-any.whl
  huggingface_hub-2.2.0-py3-none-any.whl
  idna-3.20-py3-none-any.whl
  iniconfig-2.3.1-py3-none-any.whl
  jinja2-3.1.6-py3-none-any.whl
  markdown_it_py-4.2.0-py3-none-any.whl
  markupsafe-3.0.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
  mdurl-0.1.2-py3-none-any.whl
  numpy-2.5.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
  omegaconf-2.4.0-py3-none-any.whl
  packaging-26.3-py3-none-any.whl
  pluggy-1.7.0-py3-none-any.whl
  psutil-7.2.2-cp36-abi3-manylinux2010_x86_64.manylinux_2_12_x86_64.manylinux_2_28_x86_64.whl
  pygments-2.21.0-py3-none-any.whl
  pytest-9.1.1-py3-none-any.whl
  pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
  regex-2026.9.29-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
  rich-15.0.0-py3-none-any.whl
  safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  setuptools-84.2.1-py3-none-any.whl
  shellingham-1.5.4-py2.py3-none-any.whl
  tokenizers-0.23.3-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  tqdm-4.70.1-py3-none-any.whl
  transformers-5.20.2-py3-none-any.whl
  truststore-0.10.5-py3-none-any.whl
  typer-0.27.3-py3-none-any.whl
  typing_extensions-4.16.0-py3-none-any.whl
)
mkdir -p "$WHEELS"
for w in "${WHEEL_NAMES[@]}"; do
  [ -f "$WHEELS/$w" ] || curl -fsSL --retry 5 -o "$WHEELS/$w" "$WHEELS_REPO/$w"
done

# uv is preinstalled on the image; fall back to pip if uv can't see the system-site torch
if ! uv pip install --python "$(command -v python)" --no-index --find-links "$WHEELS" \
    transformers accelerate omegaconf safetensors huggingface_hub hf_xet pytest regex; then
  echo "uv failed; retrying with pip" >&2
  pip install --no-index --find-links "$WHEELS" \
    transformers accelerate omegaconf safetensors huggingface_hub hf_xet pytest regex
fi

# vendored iris-3b: from the HF tarball (GitHub releases measured ~23 KB/s on this node);
# used via PYTHONPATH instead of a pip -e build (hatchling is not in the wheels repo)
if [ ! -d vendor/iris-3b ]; then
  curl -fsSL --retry 5 -o /home/dev/vendor-iris3b.tar.gz \
    https://huggingface.co/hasanfaesal/iris-wheels-temp/resolve/main/vendor-iris3b.tar.gz
  tar xzf /home/dev/vendor-iris3b.tar.gz
fi
git -C vendor/iris-3b checkout -q "$IRIS_COMMIT" 2>/dev/null || echo "vendor tarball has no git dir; trusting it"
export PYTHONPATH="$PWD/vendor/iris-3b/src${PYTHONPATH:+:$PYTHONPATH}"

# base weights (~12 GB fp32) and the text encoder (~8 GB), both via fast HF
mkdir -p "$MODELS"
hf download speridlabs/iris-3b --local-dir "$MODELS/iris-3b" \
  --exclude "depth/*" --exclude "upscaler/*" --exclude "turbo/*"
hf download Qwen/Qwen3-VL-4B-Instruct

python - <<'EOF'
import torch, transformers, iris3b
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
EOF
python -m pytest -q tests
echo "setup ok; before run_verification.py export HF_HOME=$HF_HOME PYTHONPATH=$PWD/vendor/iris-3b/src"
