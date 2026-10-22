# NOTES.md — Iris-3B source map (Stage 0)

Source read from `vendor/iris-3b` (speridlabs, pinned commit) and `vendor/iris-3b-turbo` (pentaCoxian fork).
All line refs are to the vendored files. `NOTES` guesses from IMPLEMENT.md are marked GUESS → REAL.

## 1. Model construction & forward

- Build: `IrisDiT(cfg.model)`; weights: meta-device build, then `model.load_state_dict(weights, strict=True, assign=True)`
  (pattern from `scripts/sample.py:92-96`). Train on bf16 base + fp32 adapters (`distill_turbo.py:175-177`).
- GUESS `model(x=, x0=, r_step=)` → REAL signature (`models/dit.py:259-266`):
  ```python
  out = model(x, t, y, capture=(), y_mask=None)   # positional (x, t, y)
  v = out.x                                        # IrisOutput.x, velocity [B, C, H, W]
  ```
  There is **no x0 input and no r input** — both must be added by our wrapper.
- `t` is model time = `1000 * shifted_sigma`, shift = 4.0 (`flow/schedule.py:7-12`, `config.py:128`).
  Floats OK at inference; base training int-truncates. `t = 1000.0` ⇔ pure noise (sigma' = 1). IMPLEMENT.md guess correct.
- Flow parameterization (`flow/transport.py:53-55`, `distill/objective.py:30-36`): `v = eps - x0`,
  `x0 = x_t - sigma' * v` (`sampling.py:130`), so at t=1000: `x0_hat = z - v`. Rollout math in IMPLEMENT.md confirmed.

## 2. Embedders (`nn/embeddings.py`)

- `s_embedder = PatchEmbedder(p*p*in_channels=768 -> 2560)`; forward takes unfolded patch vectors
  `[B, L, 768]` produced in `dit.py:289` by `F.unfold(x, 16, 16).transpose(1,2)`.
  → `x0_embedder = nn.Linear(768, 2560, bias=False)`, zero-init, summed onto `s_embedder(patches)`. GUESS correct.
- `t_embedder = TimestepEmbedder(dim, max_period=10)` (NOT 10000) → `[B, 1, D]`; forward does
  `cond = F.silu(t_emb)` (`dit.py:291-292`). Step embedding must be added BEFORE the silu:
  `cond = F.silu(t_emb + step_emb(r))`. IMPLEMENT.md formula matches.
- `y_embedder = LayerwiseTextEmbedder` (config `text_adapter: lap_blocks2`) — REQUIRES `y_mask`
  (raises maskless, `dit.py:299-304`).

## 3. Text conditioning (`text/qwen3_vl.py`, `text/base.py`)

- `enc = text_encoder.encode(prompts)` → `TextEncoding(embeddings, mask)`:
  - `embeddings`: `[B, 300, 12, 2560]` — Qwen3-VL language-stack hidden states of layers (1-based) 2,5,8,...,35,
    stacked on dim 2, padded positions zeroed (`qwen3_vl.py:180-182`).
  - `mask`: `[B, 300]` int64, 1 = real token.
- Prompt template: system+user prefix + caption + "\nassistant\n", padded to `max_length=300`.
- CFG null: `text_encoder.null("")` — empty negative prompt, same template; **cached to disk**
  under `text_encoder.null_embed_dir` (default `output/pretrained_models`). Encode once, reuse.
- Qwen3-VL-4B-Instruct is a separate frozen model (~8 GB bf16); runs under `TEXT_ENCODERS.build(cfg.text_encoder.name, ...)`.

## 4. Shared modulation cores (`nn/modulation.py`, `dit.py:118-122`)

- `IrisDiT.modulation_cores`: `nn.ModuleDict`, official config `modulation: shared_bias` → keys
  `adaln_img`, `adaln_txt`, each `nn.Linear(2560, 6*2560)`. Single-stream blocks alias `"shared" -> "img"`.
- Every block holds an unregistered `_core = (core,)` tuple pointing at the shared Linear. After wrapping a core,
  blocks keep calling the OLD linear unless you rebind — turbo's fix (`turbo nn/modulation.py:126-135`):
  ```python
  def rebind_shared_cores(model, replaced):   # replaced: id(old) -> new module
      for module in model.modules():
          core = getattr(module, "_core", None)
          if isinstance(core, tuple) and core and id(core[0]) in replaced:
              module._core = (replaced[id(core[0])],)
  ```
  `rebind_shared_cores` does NOT exist in the official package — port it into our `src/models/lora.py`.

## 5. LoRA (turbo `src/iris3b/lora.py` — port nearly verbatim)

- `LoRALinear(base_linear)`: frozen base + `adapters: ModuleDict{name: Sequential(down,up) fp32}` + `scales`
  + `active: str | None`. `active=None` = bare base (teacher). Zero-init `up` ⇒ fresh adapter is an exact no-op.
- `inject_lora(model, name, rank, alpha, targets)` wraps every `nn.Linear` under targets; idempotent (adds a slot
  to already-wrapped layers); calls `rebind_shared_cores` at the end.
- `use_adapter(model, name)` context manager sets `.active` on all LoRALinear layers.
- `save_adapter / load_adapter` (safetensors, per-layer `lora_down/lora_up`), `merge_lora_file` for export.
- DEFAULT_TARGETS = `("s_embedder", "t_embedder", "modulation_cores", "blocks", "pixel_blocks", "final_layer")`.
  NOT wrapped: `y_embedder` (frozen text adapter), `pixel_embedder` (PiT pixel proj).
- Wrapped Linear inventory: trunk attention qkv/q/k/v + proj(+x/y), attn_gate, SwiGLU w1/w3/w2, GeluMLP fc1/fc2,
  shared cores `adaln_img`/`adaln_txt`, PiT `adaln`/`compress`/`expand` + its attn/mlp, both embedders, final head.

## 6. PiT head (`dit.py:160-181`, `blocks/pit.py`)

- `pixel_embedder = PixelEmbedder(3, 16, p)` embeds the **noisy input x** per pixel (official `pixel.modulation: post`).
- PiT blocks are conditioned on `s_cond = silu(t_emb + s)` reshaped `[B*L, D]` (`dit.py:322-330`) — i.e. the PiT
  head sees x̂0 only through the trunk tokens (which include the x0 branch). IMPLEMENT.md's note is correct;
  the PiT head itself needs no new branch.
- Trunk output also gets `s = F.silu(t_emb + s)` re-fusion before the PiT stage (`dit.py:321`).

## 7. Noise schedule & teacher CFG window

- `FlowSchedule(1000, shift=4.0)`: sigma grid `1 - linspace(1.0, 0.001, 1000)`, then
  `sigma' = shift*sigma / (1 + (shift-1)*sigma)`; `t_model = 1000 * sigma'`.
- Base training samples `timestep_idx` via logit-normal (`flow/timesteps.py`). Our data-space recipe instead
  samples tau ~ Beta(2.5, 1) (pre-shift), then renoises with the SHIFTED sigma' and conditions t = 1000*sigma'
  (matches `RectifiedFlow.add_noise` semantics). Raw-vs-shifted window question settled in Stage 4.
- Windowed teacher CFG is a native mechanism: `FlowDPMSolver` applies CFG only when `lo < t' < hi`
  (`flow/solver.py:57-59`), `sample.cfg_interval` default `[0.0, 1.0]` (`config.py:501`). IMPLEMENT.md's
  "CFG only when tau <= 0.6" = interval `(0.0, 0.6)` (window on the shifted sigma, same convention as the solver).

## 8. Training-loop reference (turbo `scripts/distill_turbo.py` + `turbo iris3b/distill.py`)

DMD2-LoRA on ONE GPU, directly reusable patterns:
- bf16 frozen base, fp32 adapters; AdamW(betas=(0.0, 0.999), weight_decay=0), grad clip 1.0.
- Student LR 2.5e-5 / critic LR 5e-5, rank 64, TTUR 5:1, batch 1, `activation_checkpointing = "full"`.
- EMA over student adapter params with an `applied()` context manager; previews every N steps.
- Collapse sentinel = mean per-image pixel std of previews (collapse ~0.14, good ~0.23-0.44; stop below 0.2).
- Turbo sampling: sigmas (1, 0.938, 0.833, 0.625) from `distill.yaml`, `clean = x - sigma' * v`, no CFG.
- CRITICAL: both update functions call `loss.backward()` INSIDE the `use_adapter(...)` context — activation
  checkpointing recomputes blocks during backward and must see the same live adapter (`distill.py:20-22`).
- Teacher CFG via doubled batch `cat([uncond, cond])` (`distill.py:60-70`); IMPLEMENT.md instead runs cond and
  uncond sequentially for memory — keep IMPLEMENT.md's version.
- `Conditioning` dataclass (cond, cond_mask, null, null_mask) — reuse as-is.

## 9. Differences from IMPLEMENT.md to carry into Stages 2-6

| IMPLEMENT.md guess | Reality / decision |
|---|---|
| `model(x=, x0=, r_step=)` | Real forward is positional `(x, t, y, y_mask)`; wrapper subclass adds `x0` + `r` args |
| PiT needs no change | Confirmed (conditioning flows through `s_cond`) |
| `t=1000.0` at t=1 | Correct (shifted sigma' = 1) |
| `Linear(16*16*3, 2560)` x0 branch | Correct (768 -> 2560, zero-init, no bias) |
| RoleLoRA with 3 roles | turbo `LoRALinear` already supports N named slots + `active=None` teacher; port it |
| Rebind modulation cores | Needed; `rebind_shared_cores` must be ported (not in official pkg) |
| Windowed CFG tau <= 0.6 | Use `cfg_interval`-style gate on shifted sigma' in our teacher pass |
| 4:1 critic ratio, rank 128/64 | Our values; turbo used 5:1, rank 64/64 (fine) |
| Gen loss sum-over-pixels | Keep (differs from turbo's normalized DMD target; data-space paper style) |

## 10. Exact forward call that produces v at t=1000 (the Stage 0 self-check)

```python
import torch
from iris3b.config import inference_config
from iris3b.models.dit import IrisDiT
from iris3b.sampling import load_for_inference
from iris3b.registry import TEXT_ENCODERS

raw, weights = load_for_inference("iris-3b")            # exported dir: config.yaml + model.safetensors
cfg = inference_config(raw, [])
with torch.device("meta"):
    model = IrisDiT(cfg.model)
model.load_state_dict(weights, strict=True, assign=True)
model = model.eval().to("cuda", dtype=torch.float32)     # fp32 weights, bf16 autocast
text_encoder = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device="cuda")
enc = text_encoder.encode(["a red fox sleeping in fresh snow, golden hour"])

t = torch.full((1,), 1000.0, device="cuda")              # 1000 * sigma'(=1) = pure noise
with torch.autocast("cuda", dtype=torch.bfloat16):
    v = model(z_bf16, t, enc.embeddings, y_mask=enc.mask).x   # <- THE call: velocity at t=1000
```

## 11. Environment

- `pip install -e` blocked by PEP 668 on Arch; installed into workspace venv `.venv/`
  (CPU torch wheel; transformers/accelerate/wandb/dion intentionally omitted — text encoder needs
  transformers on the H100 only).
- Verify: `.venv/bin/python -c "import iris3b"`.

## 12. Stage 4 decisions (paper-verified against 2610.12102v1 Appendix/Eq. 8-11)

- Rollout (Eq. 8): r uniform in {0..R-1} per global batch; r no-grad steps then ONE grad step;
  z fixed at every step; t pinned to 1000.0.
- Renoise (Eq. 9): fresh independent eps + tau per loss; x_tau = (1-sigma')x0 + sigma'eps with
  sigma' = shift_sigma(tau, 4.0); model conditioned on t = 1000*sigma'.
- tau ~ Beta(2.5, 1) raw, shifted through Iris's shift, clamped to [0.02, 0.98] AFTER the shift
  (PLAN.md's convention; raw-vs-shifted settled: shift, because Iris conditions on shifted levels).
- Gen loss (Eq. 10) w(tau)=1: `(x0 * sg[v_pos - v_fake]).flatten(1).sum(1).mean()`; direction
  detached; sum over pixels per IMPLEMENT.md.
- Fake loss (Eq. 11): MSE(v_fake(x_tau, sigma', c, r), eps - x0), mean-reduction (Adam makes
  loss scale irrelevant; matches turbo's critic convention).
- CFG window: paper Sec. "CFG hyperparameter search": "a scale of 1.1 only over tau in [0.0, 0.6]"
  -> apply CFG where sigma' <= 0.6, plain v_cond above. Window is on the SHIFTED sigma' (the same
  variable that parameterizes the renoise and conditioning). cfg_scale stays 3.0 (Iris default)
  per IMPLEMENT.md; paper's 1.1 was for their SiT setup.
- Paper inits the r embedding from N(0, 0.02); we keep IMPLEMENT.md's zero-init (required by the
  zero-init identity gates).
- Tiny fixture: base weights random at std 0.2 (not untrained 0.02) and adaln_zero_init=False so
  the adaLN gates are live and the text->image path (the only CFG-relevant path) has signal.
## 13. Stage 5 decisions (prompts)

- 40-60K was speculative. What matters is per-prompt generator reuse: ~2,400 updates x batch 16
  = ~38K gen samples => 25K uniques => ~1.5 visits/prompt. Community collapse was LR-driven (5e-5),
  not prompt-count-driven. No synthetic top-up: Qwen-Image-Flash (2606.03746) reports eval-shaped
  category mixes can underperform simpler pools; strict category quotas dropped with it.
- Training pool (actual 23,926, `scripts/prepare_prompts.py` -> `data/prompts_train.jsonl`):
  flux 10,000 (long captions, of 16,912 passing), image-preferences 4,926, leffff text-rendering
  train 3,000 (~15% quoted text after mix), compbench train 2,000 (counting/spatial),
  pick-a-pic 4,000 (of 44,185 passing; strict filter dropped 14.7K). Median 28 words, max 195.
  Parti dropped from training: it is the eval set (the community LoRA trained on it; we don't).
  image_prefs is read via the datasets-server parquet endpoint with column projection (text-only;
  the full repo ships multi-GB image payloads — do NOT `load_dataset` it on a RAM-limited machine).
  Parti dropped from training: it is the eval set (the community LoRA trained on it; we don't).
- Filters: normalize -> exact dedup (train + vs eval) -> >=90% ASCII -> 5..256 Qwen tokens ->
  residue/NSFW regex (lora tags, "in the style of", artstation, URLs, JSON escapes).
  Near-dup (MinHash/embedding) skipped: exact-only; upgrade if reuse stats look bad.
- Eval: 150 Parti stratified by category + 50 leffff test = 200 held-out prompts
  (`data/prompts_eval.jsonl`), the 24 validation renders come from these.
- Text handling DECIDED: cache Qwen embeddings offline (not online): encode once on the H100,
  store unpadded [T,12,2560] bf16 + mask per prompt, pad per micro-batch, free Qwen before
  training. ~6 MB/prompt at ~100 tokens => ~140 GB NVMe for 24K; fall back to online encoding
  (Qwen bf16 ~8 GB) if NVMe is tight. Cache script belongs to the H100 stage, not stage 5.
- Buckets (IMPLEMENT.md 3.1): (1024,1024),(768,1344),(896,1152),(1344,768),(1152,896) — all
  16-aligned, ~1.0-1.03 MP. One bucket per GLOBAL batch (uniform draw); prompts carry no
  intrinsic aspect, so bucket is a batch property: `BucketBatchSampler` yields (bucket, indices).

## 14. Stage 7 results (measured on iris-dsdmd-h100-s7, 2026-10-22)

- ALL GATES PASS: teacher parity worst rel L2 err 0.0139 (tol 0.05) at t=1000/600/200; the
  bf16 base deviation is acceptable, no fp32 fallback needed. 20-step official-sampler run
  from identical noise: PSNR 29.5 dB (info only). Zero-init identity bit-exact (max abs 0),
  including teacher-unchanged-after-injection. No dead adapters after training turns.
- F = 0.128 s per 1 MP teacher forward (batch 4) -> PLAN's 4:1 critic ratio stands
  (band was <= 0.17 s).
- Ramp (full activation checkpointing, train mode): mb=2 19.6 GB / mb=4 26.9 GB / mb=6
  35.8 GB reserved. At the plan config (global 16 = mb 4 x 4): ~78 s per 4:1 update ->
  ~2,400 gen updates in ~52 h; fits the credit window with margin.
- Un-checkpointing is NOT free headroom at mb=4: --ac none, selective_layer every 2/3/4
  all OOM near the 80 GB top (PiT pixel-head activations dominate). The plan's
  "un-checkpoint trunk, keep PiT head checkpointed" recipe needs a per-stack policy (small
  code change) -> Stage 8 lever, worth <= ~17% per PLAN.
- mb=6 (global 24) measured 3.15 s/row vs mb=4's 4.85 s/row — a bigger micro-batch is
  ~35% cheaper per sample, but changes the global-batch recipe; user's call, not taken.
- transformers 5.20.2 loads Qwen3-VL fine through iris3b's encoder path (built and encoded
  during verification); the laptop-downloaded torch 2.7.1/cu126 wheels were deliberately
  NOT installed — node runs the image's torch 2.13.0+cu129.
- Node egress facts: HF ~30-460 MB/s; PyPI-family + platform S3 contexts ~20 KB/s. Deps
  installed from hasanfaesal/iris-wheels-temp via curl of wheel files (uv can't resolve
  system-site torch; pip fallback can). vendor/iris-3b used via PYTHONPATH (no hatchling
  in the wheels repo, so no pip -e).
