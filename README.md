# Iris-3B Data-Space DMD Turbo

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/hasanfaesal/iris-3b-data-space-dmd/actions/workflows/ci.yml/badge.svg)](https://github.com/hasanfaesal/iris-3b-data-space-dmd/actions/workflows/ci.yml)
[![Hugging Face Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-iris--3b--dataspace--dmd--turbo-yellow.svg)](https://huggingface.co/hasanfaesal/iris-3b-dataspace-dmd-turbo)
[![Base Model: Iris-3B](https://img.shields.io/badge/Base_Model-Iris--3B_(Pixel--Space)-orange.svg)](https://huggingface.co/speridlabs/iris-3b)
[![Paper: arXiv:2610.12102](https://img.shields.io/badge/Paper-arXiv:2610.12102-b31b1b.svg)](https://arxiv.org/abs/2610.12102)

4-step **Data-Space Distribution Matching Distillation (DMD2)** of [Iris-3B](https://huggingface.co/speridlabs/iris-3b), a 3B-parameter **pixel-space** diffusion transformer operating directly on 16×16 RGB patches without a VAE.

Instead of stepping along a hand-tuned ODE flow schedule with intermediate re-noising, the generator refines clean 1024×1024 pixel predictions directly in data space via zero-initialized patch and step-embedding branches attached to a frozen Iris-3B backbone with role-swapped LoRA adapters. Pretrained 4-step LoRA weights, training logs, and full 200-prompt evaluation artifacts are hosted on [Hugging Face](https://huggingface.co/hasanfaesal/iris-3b-dataspace-dmd-turbo).

---

## Visual Comparisons

*All panels show 4-step 1024×1024 generations from identical prompts:*  
**Left:** Official Turbo (TDM) | **Middle:** PentaCoxian Turbo (DMD2) | **Right:** Ours (Data-Space DMD2)

### Arts & Style Composition
> *"A raccoon wearing formal clothes, wearing a tophat and holding a cane. The raccoon is holding a garbage bag. Oil painting in the style of Hokusai."*

![Arts Comparison](assets/comparison_0065.png)

### Anatomy & Organic Textures
> *"an owl"*

![Owl Comparison](assets/comparison_0015.png)

### Text Rendering on Complex Objects
> *"A white ambulance with 'AMBULANCE' written in red on its back is parked on a city street, with a red cross symbol indicating its medical emergency service function."*

![Ambulance Text Comparison](assets/comparison_0160.png)

---

## Benchmark & Empirical Diagnosis

Evaluated across 200 diverse held-out prompts ([`data/prompts_eval.jsonl`](data/prompts_eval.jsonl)) at 1024×1024 resolution against the 20-step Iris-3B teacher:

| Model | Steps | SSIM vs Teacher | LPIPS vs Teacher | CLIP Score | Sharpness (Lap. Var.) | Saturation |
|---|---:|---:|---:|---:|---:|---:|
| Official Turbo (TDM) | 4 | 0.5753 | 0.5404 | 0.3145 | 0.0145 | 0.2842 |
| PentaCoxian Turbo (DMD2) | 4 | 0.4032 | 0.6294 | 0.3153 | 0.0073 | 0.2171 |
| **Ours (Data-Space DMD2)** | **4** | **0.5723** | **0.5968** | **0.3149** | **0.0019** | **0.2692** |

### Analysis of Output Softness
While the model preserves global geometry (SSIM 0.5723 vs. 0.5753 for Official Turbo) and matches prompt alignment (0.3149 CLIP score), outputs are visibly softer (Laplacian variance 0.0019 vs. 0.0145 and 0.0073). Three factors explain this behavior:

1. **Unnormalized score gradients:** Both Official Turbo (TDM) and PentaCoxian Turbo (DMD2) normalize distillation gradients per sample by $1 / \text{mean}(|x_0 - x_{0,\text{teacher}}|)$. Without this normalization, large low-frequency composition errors dominate updates and compress gradient dynamic range for high-frequency textures.
2. **Deterministic data-space refinement vs. flow re-noising:** Flow-based DMD2 re-noises intermediate states along an ODE schedule, injecting stochastic variance at every step. Pure data-space iteration refines clean $\widehat{x}_0$ deterministically from fixed noise; under reverse-KL score matching without a GAN/patch discriminator, deterministic refinement tends toward a smooth conditional mean.
3. **Training plateau:** Training stopped at 2,297 generator updates (~48 wall-clock hours on a single H100) to reserve compute for evaluation. Loss and gradient norms flattened over the final 150 updates, indicating that restoring micro-textures likely requires gradient normalization plus a perceptual or patch-discriminator regularizer rather than longer training alone.

---

## Method & Architecture

### Data-Space Iteration + DMD2
Most text-to-image diffusion models compress images through an 8× VAE, dropping thin structures and fine textures. Iris-3B removes the VAE entirely and predicts raw 16×16 RGB pixel patches. To avoid patch-boundary artifacts from few-step ODE re-noising, we adapt *Few-Step Generation via Data-Space Iteration* ([arXiv:2610.12102](https://arxiv.org/abs/2610.12102)): holding initial noise $z$ fixed at $t = 1000$ ($\sigma' = 1$), the generator iteratively refines clean pixel-space estimates:

$$\widehat{x}_0^{(0)} = \mathbf{0}, \qquad \widehat{x}_0^{(r+1)} = z - v_\theta\!\left(z, \widehat{x}_0^{(r)}, r\right), \quad r \in \{0, \dots, R-1\}$$

The student is trained with DMD2 distribution matching and backward-simulation rollouts against a frozen teacher $v_{\text{real}}$ and a dynamic fake-score critic $v_{\text{fake}}$ (4:1 TTUR, no offline teacher trajectories):

$$\nabla_\theta \mathcal{L}_{\text{DMD}} = \mathbb{E}_{z, r, \tau, \varepsilon} \left[ w(\tau) \left( v_{\text{real}}(\widehat{x}_\tau^{(r+1)}, \tau) - v_{\text{fake}}(\widehat{x}_\tau^{(r+1)}, \tau, r) \right)^\top \nabla_\theta \widehat{x}_0^{(r+1)} \right]$$

### Implementation Highlights
- **Hook-based backbone wrapper ([`src/models/dit.py`](src/models/dit.py)):** `DataSpaceIris` subclasses `IrisDiT` without forking its forward pass, injecting two zero-initialized branches (`x0_embedder` and `step_embeddings` / `fake_step_embeddings`) via forward hooks on `s_embedder` and `t_embedder`. With adapters off, the model is bit-identical to the frozen teacher.
- **Role-swapped LoRA ([`src/models/lora.py`](src/models/lora.py)):** A single bf16 base model hosts `"gen"` and `"fake"` LoRA adapters (rank 128, alpha 128) toggled via `use_adapter(model, role)`, including shared-core adaLN rebinding (`rebind_shared_cores`).
- **Rollout & Losses ([`src/engine/rollout.py`](src/engine/rollout.py), [`src/engine/losses.py`](src/engine/losses.py)):** Implements backward-simulated data-space rollouts (gradients only through the final step) and shifted-$\sigma$ DMD2 / critic objectives.
- **Trainer ([`src/engine/trainer.py`](src/engine/trainer.py)):** Single-GPU H100 training loop with 4:1 critic/generator TTUR, 200-step critic warmup, dual fast/slow EMA shadows, activation checkpointing, and collapse detection (`StdSentinel`).

---

## Quickstart (4-Step Inference)

### 1. Environment Setup
```bash
git clone https://github.com/hasanfaesal/iris-3b-data-space-dmd.git
cd iris-3b-data-space-dmd

# Install dependencies (Python >= 3.11)
pip install -e ".[eval,dev]"

# Clone the pinned Iris-3B backbone package
git clone https://github.com/speridlabs/iris-3b.git vendor/iris-3b
git -C vendor/iris-3b checkout 13fcf6a4f7fcb36561eac3daa5ad30d0b0a1822e
export PYTHONPATH="$PWD/vendor/iris-3b/src:$PYTHONPATH"
```

### 2. Download Base Weights & Generate
```bash
# Download base Iris-3B weights (excludes optional depth/upscaler/turbo folders)
huggingface-cli download speridlabs/iris-3b --local-dir models/iris-3b \
  --exclude "depth/*" --exclude "upscaler/*" --exclude "turbo/*"

# Generate a 1024x1024 image in 4 steps (auto-downloads our LoRA from Hugging Face)
python scripts/infer.py \
  --base-model models/iris-3b \
  --adapter hasanfaesal/iris-3b-dataspace-dmd-turbo \
  --prompt "an oil painting of an owl in the style of Hokusai" \
  --steps 4 \
  --out owl.png
```

To verify the pipeline on CPU without downloading model weights:
```bash
python scripts/infer.py --tiny --out /tmp/smoke.png
```

---

## Reproducing Training & Evaluation

```bash
# 1. Pre-encode training & eval prompts with Qwen3-VL-4B-Instruct
python scripts/cache_embeddings.py --checkpoint models/iris-3b --out cache

# 2. (Optional) Run real-weights parity & memory verification
python scripts/run_verification.py --checkpoint models/iris-3b --out runs/verify

# 3. Train 4-step data-space DMD2 generator (single H100 80GB)
python scripts/train.py --checkpoint models/iris-3b --cache cache --out runs/main --ac full

# 4. Export standalone safetensors LoRA + data-space branch weights
python scripts/export.py --checkpoint runs/main/checkpoints/last.pt --out exported/iris-3b-dataspace-turbo-lora.safetensors

# 5. Evaluate across the 200-prompt benchmark
python scripts/evaluate.py --base-model models/iris-3b --checkpoint runs/main/checkpoints/last.pt --eval-cache cache_eval --out runs/eval
```

### Running the Test Suite
All unit tests run on CPU using synthetic tiny fixtures ([`tests/tiny.py`](tests/tiny.py)):
```bash
PYTHONPATH="$PWD/vendor/iris-3b/src" pytest -q tests
```

---

## Repository Structure

```text
├── src/
│   ├── models/          # DataSpaceIris hook wrapper, role-swapped LoRA, meta-device loader
│   ├── engine/          # Data-space rollout, DMD2/critic losses, TTUR trainer & EMA
│   ├── data/            # Aspect-bucketed prompt dataset & cached text-embedding reader
│   └── utils/           # Atomic checkpointing & platform hooks
├── scripts/
│   ├── infer.py         # Standalone 4-step inference CLI (HF Hub or local adapter)
│   ├── train.py         # Main training entrypoint (supports --tiny CPU smoke runs)
│   ├── cache_embeddings.py
│   ├── prepare_prompts.py
│   ├── export.py        # Safetensors adapter export & merging
│   ├── evaluate.py      # Multi-model 200-prompt evaluation harness
│   ├── evaluate_baselines.py
│   ├── run_verification.py
│   ├── diagnostics/     # One-off padding, CFG, and teacher realism checks
│   └── cluster/         # H100 node provisioning, launcher, and HF upload scripts
├── tests/               # CPU unit tests for models, LoRA, rollout, losses, trainer, export, infer
├── data/                # Training and 200-prompt evaluation JSONL splits
└── assets/              # Side-by-side visual comparisons
```

---

## References

- **Data-space iteration:** *Few-Step Generation via Data-Space Iteration* ([arXiv:2610.12102](https://arxiv.org/abs/2610.12102))
- **Base architecture:** *Iris-3B: Going Beyond the Latent with Pixel-Space Diffusion Training, Conversion and Fine-Tuning* ([arXiv:2610.09450](https://arxiv.org/abs/2610.09450))
- **Distillation framework:** *Improved Distribution Matching Distillation for Fast Image Synthesis (DMD2)* ([arXiv:2405.14867](https://arxiv.org/abs/2405.14867))

---

## Citation

```bibtex
@misc{faesal2026iris3bdataspace,
  author = {Faesal, Hasan},
  title = {Iris-3B Data-Space DMD Turbo: 4-Step Distillation of Pixel-Space Diffusion Transformers},
  year = {2026},
  howpublished = {\url{https://github.com/hasanfaesal/iris-3b-data-space-dmd}},
  note = {Pretrained weights: \url{https://huggingface.co/hasanfaesal/iris-3b-dataspace-dmd-turbo}}
}
```
