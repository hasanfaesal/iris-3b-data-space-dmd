# Iris-3B Data-Space DMD

[Weights (Hugging Face)](https://huggingface.co/hasanfaesal/iris-3b-dataspace-dmd-turbo) · [Base Model (Iris-3B)](https://huggingface.co/speridlabs/iris-3b) · [License](LICENSE)

4-step data-space Distribution Matching Distillation (DMD2) of [Iris-3B](https://huggingface.co/speridlabs/iris-3b), a 3B-parameter pixel-space diffusion transformer that operates directly on 16×16 RGB patches without a VAE.

Adapting *Few-Step Generation via Data-Space Iteration* ([arXiv:2610.12102](https://arxiv.org/abs/2610.12102)), the generator holds the initial noise fixed and iteratively refines clean 1024×1024 pixel estimates directly in data space, avoiding intermediate ODE re-noising. Zero-initialized patch and step-embedding branches are attached to a frozen Iris-3B backbone with role-swapped LoRA adapters. Pretrained LoRA weights, training logs, and 200-prompt evaluation outputs are hosted on [Hugging Face](https://huggingface.co/hasanfaesal/iris-3b-dataspace-dmd-turbo).

## Results

All models are evaluated at 4 steps and 1024×1024 resolution across 200 held-out prompts ([`data/prompts_eval.jsonl`](data/prompts_eval.jsonl)), with SSIM and LPIPS measured against the 20-step Iris-3B teacher:

| Model | Steps | SSIM vs. Teacher (↑) | LPIPS vs. Teacher (↓) | CLIP Score (↑) | Laplacian Var. | Saturation |
|---|---:|---:|---:|---:|---:|---:|
| Official Turbo (TDM) | 4 | 0.5753 | 0.5404 | 0.3145 | 0.0145 | 0.2842 |
| PentaCoxian Turbo (DMD2) | 4 | 0.4032 | 0.6294 | 0.3153 | 0.0073 | 0.2171 |
| Ours (Data-Space DMD2) | 4 | 0.5723 | 0.5968 | 0.3149 | 0.0019 | 0.2692 |

### Qualitative Samples

4-step 1024×1024 generations from identical prompts. Left: Official Turbo (TDM) · Middle: PentaCoxian Turbo (DMD2) · Right: Ours (Data-Space DMD2).

> *"A raccoon wearing formal clothes, wearing a tophat and holding a cane. The raccoon is holding a garbage bag. Oil painting in the style of Hokusai."*

![Hokusai raccoon comparison](assets/comparison_0065.png)

> *"an owl"*

![Owl comparison](assets/comparison_0015.png)

> *"A white ambulance with 'AMBULANCE' written in red on its back is parked on a city street, with a red cross symbol indicating its medical emergency service function."*

![Ambulance comparison](assets/comparison_0160.png)

## Observations & Limitations

Our model matches Official Turbo on global structure (SSIM 0.5723 vs. 0.5753) and prompt alignment (CLIP 0.3149 vs. 0.3145), but produces noticeably softer high-frequency textures (Laplacian variance 0.0019 vs. 0.0145). Training ran for 2,297 generator updates (~48 hours on a single H100 80GB), with loss and gradient norms flattening over the final ~150 updates—suggesting longer training alone is unlikely to recover sharpness.

Two hypotheses are worth ablating next:

1. **Per-sample gradient normalization:** Both Official Turbo (TDM) and PentaCoxian Turbo (DMD2) normalize distillation gradients per sample by $1 / \text{mean}(|x_0 - x_{0,\text{teacher}}|)$. Without it, coarse structural discrepancies dominate early and mid-training updates at the expense of high-frequency texture gradients.
2. **Deterministic refinement under reverse-KL:** Flow-based DMD2 re-noises intermediate states along an ODE schedule, whereas pure data-space iteration refines $\widehat{x}_0$ deterministically from fixed noise $z$. Under reverse-KL score matching without a GAN or patch discriminator, deterministic refinement tends toward a smoother conditional mean.

## Method

Holding the source noise $z$ fixed at $t = 1000$ ($\sigma' = 1$), the generator iteratively refines clean pixel-space estimates:

$$\widehat{x}_0^{(0)} = \mathbf{0}, \qquad \widehat{x}_0^{(r+1)} = z - v_\theta\!\left(z, \widehat{x}_0^{(r)}, r\right), \quad r \in \{0, \dots, R-1\}$$

Training uses DMD2 distribution matching with backward-simulated rollouts (gradients only through the final step) against the frozen teacher $v_{\text{real}}$ and a dynamic fake-score critic $v_{\text{fake}}$ (4:1 TTUR, no offline teacher trajectories):

$$\nabla_\theta \mathcal{L}_{\text{DMD}} = \mathbb{E}_{z, r, \tau, \varepsilon} \left[ w(\tau) \left( v_{\text{real}}(\widehat{x}_\tau^{(r+1)}, \tau) - v_{\text{fake}}(\widehat{x}_\tau^{(r+1)}, \tau, r) \right)^\top \nabla_\theta \widehat{x}_0^{(r+1)} \right]$$

To avoid forking the upstream backbone, [`DataSpaceIris`](src/models/dit.py) subclasses `IrisDiT` and injects zero-initialized `x0_embedder` and step-embedding branches via forward hooks on `s_embedder` and `t_embedder`, preserving bit-exact teacher parity when adapters are disabled. A single bf16 backbone hosts both `"gen"` and `"fake"` LoRA adapters (rank 128, alpha 128) toggled through [`use_adapter`](src/models/lora.py), including shared adaLN core rebinding. Rollout, shifted-$\sigma$ DMD2/critic losses, and the single-GPU training loop (200-step critic warmup, dual EMA, activation checkpointing) live in [`src/engine/`](src/engine/).

## Inference

```bash
git clone https://github.com/hasanfaesal/iris-3b-data-space-dmd.git
cd iris-3b-data-space-dmd

pip install -e ".[eval,dev]"

# Pinned Iris-3B backbone
git clone https://github.com/speridlabs/iris-3b.git vendor/iris-3b
git -C vendor/iris-3b checkout 13fcf6a4f7fcb36561eac3daa5ad30d0b0a1822e
export PYTHONPATH="$PWD/vendor/iris-3b/src:$PYTHONPATH"

# Download base Iris-3B weights (excludes optional depth/upscaler/turbo folders)
huggingface-cli download speridlabs/iris-3b --local-dir models/iris-3b \
  --exclude "depth/*" --exclude "upscaler/*" --exclude "turbo/*"

# Generate a 1024x1024 image in 4 steps (downloads our LoRA from Hugging Face)
python scripts/infer.py \
  --base-model models/iris-3b \
  --adapter hasanfaesal/iris-3b-dataspace-dmd-turbo \
  --prompt "an oil painting of an owl in the style of Hokusai" \
  --steps 4 \
  --out owl.png
```

CPU smoke test (no weights required):

```bash
python scripts/infer.py --tiny --out /tmp/smoke.png
```

## Training & Evaluation

```bash
# 1. Pre-encode training and eval prompts with Qwen3-VL-4B-Instruct
python scripts/cache_embeddings.py --checkpoint models/iris-3b --out cache

# 2. (Optional) Verify teacher parity and memory footprint on real weights
python scripts/run_verification.py --checkpoint models/iris-3b --out runs/verify

# 3. Train 4-step data-space DMD2 generator (single H100 80GB)
python scripts/train.py --checkpoint models/iris-3b --cache cache --out runs/main --ac full

# 4. Export standalone safetensors LoRA + data-space branch weights
python scripts/export.py --checkpoint runs/main/checkpoints/last.pt --out exported/iris-3b-dataspace-turbo-lora.safetensors

# 5. Evaluate on the 200-prompt benchmark
python scripts/evaluate.py --base-model models/iris-3b --checkpoint runs/main/checkpoints/last.pt --eval-cache cache_eval --out runs/eval
```

Unit tests run on CPU using synthetic tiny fixtures ([`tests/tiny.py`](tests/tiny.py)):

```bash
PYTHONPATH="$PWD/vendor/iris-3b/src" pytest -q tests
```

## References

- *Few-Step Generation via Data-Space Iteration* ([arXiv:2610.12102](https://arxiv.org/abs/2610.12102))
- *Iris-3B: Going Beyond the Latent with Pixel-Space Diffusion Training, Conversion and Fine-Tuning* ([arXiv:2610.09450](https://arxiv.org/abs/2610.09450))
- *Improved Distribution Matching Distillation for Fast Image Synthesis (DMD2)* ([arXiv:2405.14867](https://arxiv.org/abs/2405.14867))
