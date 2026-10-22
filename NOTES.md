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