"""Stage 7: real-weights verification on the H100 (or `--tiny` for a CPU dry run of this script).

    python scripts/run_verification.py --checkpoint /home/dev/models/iris-3b --out runs/verify
    python scripts/run_verification.py --tiny --out /tmp/verify

Checks, in order (exit code 1 if a gate fails; results land in <out>/verification.json):
  1. teacher parity: our bf16 base (adapters off) vs the official fp32-under-autocast model,
     velocity at several t plus a short official-sampler run from identical noise
  2. zero-init identity: fresh gen/fake adapters + branches == teacher, bit-exact
  3. F: seconds per 1 MP teacher forward (picks the critic ratio, PLAN.md table)
  4. VRAM / step-time ramp over micro-batch sizes with the real Trainer
  5. every adapter B matrix moved off zero (no dead adapters)
"""

import argparse
import contextlib
import copy
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from iris3b.sampling import generate  # noqa: E402
from iris3b.text.base import TextEncoding  # noqa: E402

from src.engine.trainer import TrainConfig, Trainer  # noqa: E402
from src.models.lora import inject_lora, lora_layers, use_adapter  # noqa: E402

PROMPT = "a red fox sleeping in fresh snow, golden hour, detailed fur, shallow depth of field"
GATES: dict[str, bool] = {}


class CachedEncoder:
    """Duck-types TextEncoder for iris3b.sampling.generate, serving pre-encoded tensors."""

    def __init__(self, y, mask, null, null_mask):
        self.y, self.mask, self.null_enc = y, mask, TextEncoding(null, null_mask)

    def encode(self, prompts):
        return TextEncoding(self.y.expand(len(prompts), *self.y.shape[1:]), self.mask.expand(len(prompts), -1))

    def null(self, negative_prompt=""):
        return self.null_enc


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def rel_err(ref, x):
    return ((ref.float() - x.float()).norm() / ref.float().norm()).item()


def save_png(x, path):
    from PIL import Image

    img = ((x.float().clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(path)


def gate(name, ok, detail=""):
    GATES[name] = bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def build_real(args, device):
    import iris3b.text  # noqa: F401  (registers encoders)
    from iris3b.config import inference_config
    from iris3b.models.dit import IrisDiT
    from iris3b.registry import TEXT_ENCODERS
    from iris3b.sampling import load_for_inference

    from src.models.loading import load_dataspace_iris

    raw, weights = load_for_inference(args.checkpoint)
    cfg = inference_config(raw, [])
    with torch.device("meta"):
        ref = IrisDiT(cfg.model)
    ref.load_state_dict(weights, strict=True, assign=True)
    ref = ref.eval().to(device=device, dtype=torch.float32)  # exactly sample.py's model
    model, _ = load_dataspace_iris(args.checkpoint, device, weights=weights, raw=raw)
    del weights
    gc.collect()

    enc = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=device)
    e, n = enc.encode([PROMPT]), enc.null("")
    y, mask = e.embeddings.to(device), e.mask.to(device)
    null, null_mask = n.embeddings.to(device), n.mask.to(device)
    del enc  # Qwen is only needed for these two encodings
    gc.collect()
    torch.cuda.empty_cache()
    return ref, model, y, mask, null, null_mask, (1024, 1024)


def build_tiny():
    from tiny import dataspace_iris, tiny_cond

    model = dataspace_iris()
    y, mask = tiny_cond(1)
    null, null_mask = tiny_cond(1)
    return copy.deepcopy(model), model, y, mask, null, null_mask, (16, 16)


@torch.no_grad()
def check_parity(ref, model, y, mask, null, null_mask, hw, device, args, out):
    h, w = hw
    z = torch.randn(2, 3, h, w, generator=torch.Generator().manual_seed(0)).to(device)
    ref_dtype = next(ref.parameters()).dtype
    worst, teacher_v = 0.0, {}
    for t in (1000.0, 600.0, 200.0):
        tv = torch.full((2,), t, device=device)
        with autocast(device):
            v_ref = ref(z, tv, y.expand(2, *y.shape[1:]).to(ref_dtype), y_mask=mask.expand(2, -1)).x
            v = model(z, tv, y.expand(2, *y.shape[1:]), y_mask=mask.expand(2, -1)).x
        e = rel_err(v_ref, v)
        worst = max(worst, e)
        teacher_v[t] = v.float().clone()
        print(f"  parity t={t:.0f}: rel L2 err {e:.4f}")
    gate("teacher velocity parity", worst < args.tol, f"worst rel err {worst:.4f} (tol {args.tol})")

    # the official sampler path from identical noise: informational (trajectories amplify small diffs)
    enc = CachedEncoder(y, mask, null, null_mask)
    noise = torch.randn(1, 3, h, w, generator=torch.Generator().manual_seed(1)).to(device)
    imgs = []
    for m in (ref, model):
        with autocast(device):
            imgs.append(generate(m, enc, [PROMPT], h, w, steps=args.sample_steps, noise=noise, device=device))
    mse = (imgs[0] - imgs[1]).float().pow(2).mean().item()
    psnr = 10 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()
    print(f"  official sampler, {args.sample_steps} steps: PSNR(ref, ours) = {psnr:.1f} dB (info)")
    for name, img in zip(("ref", "ours"), imgs):
        save_png(img[0], out / f"parity_{name}.png")
    return teacher_v, {"worst_rel_err": worst, "sampler_psnr_db": psnr}


@torch.no_grad()
def check_identity(model, y, mask, hw, device, teacher_v_pre, ranks):
    inject_lora(model, "gen", ranks[0])
    inject_lora(model, "fake", ranks[1])
    h, w = hw
    z = torch.randn(2, 3, h, w, generator=torch.Generator().manual_seed(0)).to(device)
    x0 = torch.randn_like(z)  # garbage x0: the zero-init branch must ignore it
    yy, mm = y.expand(2, *y.shape[1:]), mask.expand(2, -1)
    ok, worst = True, 0.0
    for t in (1000.0, 600.0, 200.0):
        tv = torch.full((2,), t, device=device)
        with autocast(device):
            with use_adapter(model, "off"):
                teacher = model(z, tv, yy, y_mask=mm).x
            with use_adapter(model, "gen"):
                gen = model(z, tv, yy, x0=x0, r=2, y_mask=mm).x
            with use_adapter(model, "fake"):
                fake = model(z, tv, yy, r=2, y_mask=mm).x
        for out in (gen, fake):
            worst = max(worst, (out.float() - teacher.float()).abs().max().item())
            ok &= torch.equal(out, teacher)
        # injecting adapters must not change the teacher itself
        worst = max(worst, (teacher.float() - teacher_v_pre[t]).abs().max().item())
        ok &= torch.equal(teacher.float(), teacher_v_pre[t])
    gate("zero-init identity (gen == fake == teacher == pre-injection teacher)", ok, f"max abs diff {worst:.3g}")
    return {"max_abs_diff": worst}


@torch.no_grad()
def measure_F(model, y, mask, hw, device, batch=4, reps=3):
    """Seconds per single teacher forward at ~1 MP (PLAN.md's F)."""
    h, w = hw
    z = torch.randn(batch, 3, h, w, device=device)
    tv = torch.full((batch,), 600.0, device=device)
    yy, mm = y.expand(batch, *y.shape[1:]), mask.expand(batch, -1)
    model.eval()
    with use_adapter(model, "off"), autocast(device):
        model(z, tv, yy, y_mask=mm)  # warmup
        sync(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(z, tv, yy, y_mask=mm)
        sync(device)
    F = (time.perf_counter() - t0) / reps / batch
    ratio = "4:1" if F <= 0.17 else "2:1" if F <= 0.25 else "2:1 + reuse student rollouts for one critic turn"
    print(f"  F = {F:.3f} s per forward -> PLAN.md critic ratio: {ratio}")
    return {"F_seconds": F, "critic_ratio": ratio}


def ramp(model, y, mask, null, null_mask, hw, device, args):
    """Real Trainer updates at growing micro-batch; records peak VRAM and step time."""
    model.train()
    model.activation_checkpointing = args.ac
    model.ac_selective_every = args.ac_every
    cfg = TrainConfig(micro_batch=args.mbs[0], num_steps=4, ema_decay=0.999)
    trainer = Trainer(model, null, null_mask, cfg, device=device, autocast_dtype=torch.bfloat16 if device.type == "cuda" else None)
    h, w = hw
    results = []
    for mb in args.mbs:
        trainer.cfg.micro_batch = mb
        rows = mb * 4  # one micro-batch per r (global batch = 4 micro-batches, r = k mod 4)
        batch = {"y": y.expand(rows, *y.shape[1:]), "y_mask": mask.expand(rows, -1), "hw": (h, w)}
        row = {"micro_batch": mb, "rows": rows, "ac": args.ac}
        try:
            trainer.fake_update(batch)  # warmup: allocator, optimizer state, kernels
            trainer.gen_update(batch)
            sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            repeats = getattr(args, "repeats", 3)
            t_fake, t_gen = 0.0, 0.0
            for _ in range(repeats):
                t0 = time.perf_counter()
                trainer.fake_update(batch)
                sync(device)
                t1 = time.perf_counter()
                trainer.gen_update(batch)
                sync(device)
                t2 = time.perf_counter()
                t_fake += (t1 - t0)
                t_gen += (t2 - t1)
            fake_s, gen_s = t_fake / repeats, t_gen / repeats
            fake_ps, gen_ps = fake_s / rows, gen_s / rows
            row.update(
                peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
                peak_reserved_gb=torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0,
                fake_s=fake_s, gen_s=gen_s,
                fake_ps=fake_ps, gen_ps=gen_ps,
                repeats=repeats,
                # one generator update at ratio 4:1 for a global batch of 16
                projected_update_s=16 * (4 * fake_ps + gen_ps),
            )
        except torch.OutOfMemoryError:
            row["oom"] = True
            results.append(row)
            print(f"  mb={mb}: OOM", flush=True)
            break
        results.append(row)
        print("  mb={micro_batch}: peak {peak_reserved_gb:.1f} GB reserved ({peak_allocated_gb:.1f} allocated), "
              "fake {fake_s:.1f}s gen {gen_s:.1f}s per {rows} rows, ~{projected_update_s:.0f}s per 4:1 update".format(**row), flush=True)
    dead = {name: [n for n, m in lora_layers(model).items() if m.adapters[name][1].weight.abs().sum() == 0]
            for name in ("gen", "fake")}
    gate("no dead adapters after ramp", not any(dead.values()), f"dead: { {k: v[:3] for k, v in dead.items()} }")
    ok = [r for r in results if not r.get("oom")]
    gate("ramp ran at least one micro-batch without OOM", bool(ok))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/home/dev/models/iris-3b", help="exported dir: config.yaml + model.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true", help="CPU dry run on the tiny fixture (logic check only)")
    ap.add_argument("--tol", type=float, default=0.05, help="max relative L2 velocity error, bf16 vs fp32 teacher")
    ap.add_argument("--sample-steps", type=int, default=20)
    ap.add_argument("--mbs", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--repeats", type=int, default=3, help="number of timed repeats per micro-batch")
    ap.add_argument("--ac", default="full", choices=["none", "full", "selective_op", "selective_layer"])
    ap.add_argument("--ac-every", type=int, default=2)
    ap.add_argument("--ranks", type=int, nargs=2, default=[128, 64], metavar=("GEN", "FAKE"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.tiny else "cuda")
    if args.tiny:
        args.sample_steps, args.mbs, args.ranks = 4, [1, 2], [8, 4]

    ref, model, y, mask, null, null_mask, hw = build_tiny() if args.tiny else build_real(args, device)
    report = {}
    teacher_v, report["parity"] = check_parity(ref, model, y, mask, null, null_mask, hw, device, args, out)
    del ref  # fp32 reference no longer needed; free ~12 GB before the memory measurements
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report["identity"] = check_identity(model, y, mask, hw, device, teacher_v, args.ranks)
    report["F"] = measure_F(model, y, mask, hw, device)
    report["ramp"] = ramp(model, y, mask, null, null_mask, hw, device, args)
    report["gates"] = GATES
    (out / "verification.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out / 'verification.json'}")
    sys.exit(0 if all(GATES.values()) else 1)


if __name__ == "__main__":
    main()
