"""High-throughput multi-model evaluation and benchmarking suite.

Evaluates and compares:
1. Teacher Baseline (20-step FlowDPMSolver)
2. Our Data-Space Generator (Fast EMA, 4-step)
3. Our Data-Space Generator (Slow EMA, 4-step)
4. Our Data-Space Generator (Mid-checkpoint, 4-step, optional)
5. Pentacoxian DMD2 LoRA (4-step, optional)
6. Official Iris-3B Turbo (4-step, optional)

Leverages 80GB VRAM batching, in-memory adapter/model swapping, and asynchronous
CPU workers for metrics and image writing.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "vendor" / "iris-3b"),
    str(ROOT / "vendor" / "iris-3b-turbo"),
]

from src.data.embed_cache import EmbedCache, load_null
from src.engine.rollout import rollout_dataspace
from src.models.loading import load_dataspace_iris
from src.models.lora import inject_lora, use_adapter
from src.utils.checkpoint import load_checkpoint


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Pure numpy/scipy-free mean SSIM on [0, 1] RGB images (H, W, 3)."""
    # Convert to grayscale luminance
    y1 = 0.2989 * img1[..., 0] + 0.5870 * img1[..., 1] + 0.1140 * img1[..., 2]
    y2 = 0.2989 * img2[..., 0] + 0.5870 * img2[..., 1] + 0.1140 * img2[..., 2]

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    # Block-averaged statistics (8x8 blocks for speed & stability)
    h, w = y1.shape
    bh, bw = h // 8, w // 8
    if bh == 0 or bw == 0:
        return 1.0

    y1 = y1[: bh * 8, : bw * 8].reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
    y2 = y2[: bh * 8, : bw * 8].reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3).reshape(-1, 64)

    mu1 = y1.mean(axis=1)
    mu2 = y2.mean(axis=1)
    sigma1_sq = y1.var(axis=1)
    sigma2_sq = y2.var(axis=1)
    sigma12 = ((y1 - mu1[:, None]) * (y2 - mu2[:, None])).mean(axis=1)

    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())


class OptionalLPIPS:
    """Gracefully loads lpips if installed, else returns None."""

    def __init__(self, device: torch.device):
        self.loss_fn = None
        self.device = device
        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
            self.loss_fn.eval()
            print("LPIPS (alex) loaded successfully.", flush=True)
        except Exception:
            self.loss_fn = None

    def compute(self, img1_t: torch.Tensor, img2_t: torch.Tensor) -> float | None:
        if self.loss_fn is None:
            return None
        # img1_t, img2_t: [3, H, W] in [0, 1] -> map to [-1, 1] [1, 3, H, W]
        x1 = (img1_t.unsqueeze(0).to(self.device).float() * 2.0) - 1.0
        x2 = (img2_t.unsqueeze(0).to(self.device).float() * 2.0) - 1.0
        with torch.no_grad():
            dist = self.loss_fn(x1, x2)
        return float(dist.mean().item())


class OptionalCLIPScore:
    """Gracefully loads CLIPModel via transformers if available."""

    def __init__(self, device: torch.device, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = None
        self.processor = None
        self.device = device
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.model.eval()
            print(f"CLIP Score evaluator ({model_name}) loaded successfully.", flush=True)
        except Exception as e:
            self.model = None
            self.processor = None

    def compute(self, img_pil: Image.Image, prompt: str) -> float | None:
        if self.model is None or self.processor is None:
            return None
        try:
            inputs = self.processor(text=[prompt], images=img_pil, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                score = (image_embeds @ text_embeds.T).squeeze().item()
            return float(score)
        except Exception:
            return None


def compute_image_metrics(
    img_np: np.ndarray,
    teacher_np: np.ndarray | None = None,
    img_pil: Image.Image | None = None,
    prompt: str | None = None,
    lpips_eval: OptionalLPIPS | None = None,
    clip_eval: OptionalCLIPScore | None = None,
) -> dict:
    """Compute quality diagnostics on [0, 1] RGB numpy array (H, W, 3)."""
    mean = float(img_np.mean())
    std = float(img_np.std())
    clip_black = float((img_np < 0.02).mean())
    clip_white = float((img_np > 0.98).mean())

    # HSV Saturation
    max_c = img_np.max(axis=-1)
    min_c = img_np.min(axis=-1)
    delta = max_c - min_c
    sat = np.zeros_like(max_c)
    mask = max_c > 1e-5
    sat[mask] = delta[mask] / max_c[mask]
    mean_sat = float(sat.mean())

    # Sharpness via Laplacian variance
    gray = 0.2989 * img_np[..., 0] + 0.5870 * img_np[..., 1] + 0.1140 * img_np[..., 2]
    lap = (
        np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
        - 4.0 * gray
    )
    lap_var = float(lap.var())

    res = {
        "pixel_mean": round(mean, 4),
        "pixel_std": round(std, 4),
        "mean_saturation": round(mean_sat, 4),
        "laplacian_var": round(lap_var, 6),
        "clip_black_pct": round(clip_black * 100, 2),
        "clip_white_pct": round(clip_white * 100, 2),
    }

    if teacher_np is not None:
        mae = float(np.abs(img_np - teacher_np).mean())
        psnr = float(10 * np.log10(1.0 / max(1e-6, ((img_np - teacher_np) ** 2).mean())))
        ssim_val = compute_ssim(img_np, teacher_np)
        res["mae_vs_teacher"] = round(mae, 4)
        res["psnr_vs_teacher"] = round(psnr, 2)
        res["ssim_vs_teacher"] = round(ssim_val, 4)

        if lpips_eval is not None:
            t1 = torch.from_numpy(img_np).permute(2, 0, 1)
            t2 = torch.from_numpy(teacher_np).permute(2, 0, 1)
            lp_dist = lpips_eval.compute(t1, t2)
            if lp_dist is not None:
                res["lpips_vs_teacher"] = round(lp_dist, 4)

    if clip_eval is not None and img_pil is not None and prompt:
        c_score = clip_eval.compute(img_pil, prompt)
        if c_score is not None:
            res["clip_score"] = round(c_score, 4)

    return res


def save_grid(images: list[np.ndarray], labels: list[str], path: Path) -> None:
    """Compose a horizontal side-by-side comparison strip with labels."""
    from PIL import ImageDraw

    h, w, _ = images[0].shape
    header_h = 32
    grid_img = Image.new("RGB", (w * len(images), h + header_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(grid_img)

    for i, (img, label) in enumerate(zip(images, labels, strict=True)):
        im_pil = Image.fromarray((img * 255.0).round().astype(np.uint8))
        grid_img.paste(im_pil, (i * w, header_h))
        draw.text((i * w + 10, 8), label, fill=(240, 240, 240))

    grid_img.save(path)


def load_prompts(prompts_file: Path, limit: int | None = None) -> list[dict]:
    prompts = []
    with open(prompts_file) as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item.setdefault("id", f"{idx:04d}")
            item.setdefault("category", "general")
            prompts.append(item)
    return prompts


def sample_teacher(
    model, z_batch: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
    null: torch.Tensor, null_mask: torch.Tensor, steps: int = 20, cfg_scale: float = 3.5, shift: float = 3.0
) -> torch.Tensor:
    """Run 20-step FlowDPMSolver with CFG."""
    from iris3b.flow.solver import FlowDPMSolver

    b = z_batch.shape[0]
    # Match sequence length between prompt embeddings and null embeddings
    seq_len = y.shape[1]
    if null.shape[1] != seq_len:
        if null.shape[1] > seq_len:
            null = null[:, :seq_len]
            null_mask = null_mask[:, :seq_len]
        else:
            pad_len = seq_len - null.shape[1]
            pad_emb = torch.zeros(1, pad_len, *null.shape[2:], dtype=null.dtype, device=null.device)
            pad_m = torch.zeros(1, pad_len, dtype=null_mask.dtype, device=null_mask.device)
            null = torch.cat([null, pad_emb], dim=1)
            null_mask = torch.cat([null_mask, pad_m], dim=1)

    null_expanded = null.expand(b, *null.shape[1:])
    cfg_mask = torch.cat([null_mask.expand(b, -1), mask])

    def model_fn(x, t, cond, y=y, mask=mask, cfg_mask=cfg_mask):
        m = mask if cond is y else cfg_mask
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16), use_adapter(model, "off"):
            return model(x, t, cond, y_mask=m).x.float()

    solver = FlowDPMSolver(model_fn, num_timesteps=1000, cfg_scale=cfg_scale, prediction="v")
    sample = solver.sample(
        z_batch, y, null_expanded if cfg_scale != 1.0 else None, steps=steps, order=2, shift=shift
    )
    return sample.clamp(-1, 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="/home/dev/models/iris-3b", help="Path to base Iris-3B checkpoint")
    ap.add_argument("--checkpoint", default="checkpoints/last.pt", help="Path to our trainer checkpoint")
    ap.add_argument("--mid-checkpoint", default=None, help="Path to mid-run checkpoint (e.g. checkpoints/gen001500.pt)")
    ap.add_argument("--eval-prompts", default="data/prompts_eval.jsonl", help="Evaluation prompts jsonl")
    ap.add_argument("--eval-cache", default="/home/dev/cache_eval", help="Directory with pre-cached eval embeddings")
    ap.add_argument("--out", default="runs/benchmark_eval", help="Output directory")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for parallel forward passes")
    ap.add_argument("--steps", type=int, default=4, help="Data-space rollout steps for our model (r=steps-1)")
    ap.add_argument("--hw", type=int, nargs=2, default=[1024, 1024], help="Image height and width")
    ap.add_argument("--seed", type=int, default=42, help="Evaluation random seed")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of prompts evaluated")
    ap.add_argument("--skip-teacher", action="store_true", help="Skip 20-step teacher evaluation")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    grids_dir = out_dir / "grids"
    images_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Prompts
    prompts_meta = load_prompts(Path(args.eval_prompts), limit=args.limit)
    num_prompts = len(prompts_meta)
    print(f"Loaded {num_prompts} evaluation prompts from {args.eval_prompts}")

    # 2. Embeddings (from cache or encode once)
    eval_cache_path = Path(args.eval_cache)
    if eval_cache_path.exists() and (eval_cache_path / "index.json").exists():
        print(f"Loading cached embeddings from {eval_cache_path}...")
        cache = EmbedCache(eval_cache_path)
        null_emb, null_mask = load_null(eval_cache_path)
        null_emb, null_mask = null_emb.to(device), null_mask.to(device)

        def get_batch_text(indices: list[int]):
            y, mask = cache.get(indices)
            return y.to(device, dtype=torch.bfloat16), mask.to(device)
    else:
        print(f"Embedding cache not found at {eval_cache_path}. Encoding with Qwen...")
        import iris3b.text  # noqa: F401
        from iris3b.config import inference_config
        from iris3b.registry import TEXT_ENCODERS
        from iris3b.sampling import load_for_inference

        raw_cfg, _ = load_for_inference(args.base_model)
        cfg_inf = inference_config(raw_cfg, [])
        enc = TEXT_ENCODERS.build(cfg_inf.text_encoder.name, cfg_inf.text_encoder, device=device)

        texts = [p["prompt"] for p in prompts_meta]
        all_y, all_mask = [], []
        enc_bs = 16
        for s in range(0, len(texts), enc_bs):
            res = enc.encode(texts[s : s + enc_bs])
            all_y.append(res.embeddings.to(torch.bfloat16).cpu())
            all_mask.append(res.mask.cpu())
        null_res = enc.null("")
        null_emb = null_res.embeddings.to(device, dtype=torch.bfloat16)
        null_mask = null_res.mask.to(device)
        del enc
        if device.type == "cuda":
            torch.cuda.empty_cache()

        all_y_cat = torch.cat(all_y, dim=0)
        all_mask_cat = torch.cat(all_mask, dim=0)

        def get_batch_text(indices: list[int]):
            return all_y_cat[indices].to(device), all_mask_cat[indices].to(device)

    # 3. Load Model
    print(f"Loading base DataSpaceIris from {args.base_model}...")
    model, _ = load_dataspace_iris(args.base_model)
    model.eval().to(device)

    # Inject LoRA for "gen"
    print("Injecting LoRA adapters...")
    inject_lora(model, "gen", rank=64, alpha=64.0)

    # Load checkpoint states
    print(f"Loading checkpoint {args.checkpoint}...")
    ckpt_main = load_checkpoint(args.checkpoint)
    fast_ema_state = ckpt_main["ema"]
    slow_ema_state = ckpt_main.get("ema_slow", fast_ema_state)

    mid_state = None
    if args.mid_checkpoint and Path(args.mid_checkpoint).exists():
        print(f"Loading mid-checkpoint {args.mid_checkpoint}...")
        ckpt_mid = load_checkpoint(args.mid_checkpoint)
        mid_state = ckpt_mid.get("ema", ckpt_mid.get("gen"))

    # Helper to apply parameter dict to model
    def set_gen_weights(weights_dict: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in weights_dict:
                    param.copy_(weights_dict[name])

    # Models to benchmark
    models_to_run = []
    if not args.skip_teacher:
        models_to_run.append("teacher")
    models_to_run.append("our_fast_ema")
    models_to_run.append("our_slow_ema")
    if mid_state is not None:
        models_to_run.append("our_mid_ckpt")

    print(f"Models to evaluate: {models_to_run}")

    # Optional evaluators (LPIPS and CLIP Score)
    lpips_eval = OptionalLPIPS(device)
    clip_eval = OptionalCLIPScore(device)

    executor = ThreadPoolExecutor(max_workers=6)
    futures = []
    metrics_by_model: dict[str, list[dict]] = {m: [] for m in models_to_run}

    h, w = args.hw
    total_batches = (num_prompts + args.batch_size - 1) // args.batch_size
    t_start = time.time()

    # Iterate over batches
    for b_idx in range(total_batches):
        start_idx = b_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, num_prompts)
        batch_indices = list(range(start_idx, end_idx))
        cur_b = len(batch_indices)
        batch_meta = [prompts_meta[i] for i in batch_indices]

        # Get Text conditioning
        y, mask = get_batch_text(batch_indices)

        # Deterministic noise for all models
        g = torch.Generator(device=device).manual_seed(args.seed + b_idx)
        z = torch.randn(cur_b, 3, h, w, device=device, generator=g)

        batch_outputs: dict[str, np.ndarray] = {}

        # 1. Teacher
        if "teacher" in models_to_run:
            t0 = time.time()
            sample_t = sample_teacher(model, z, y, mask, null_emb, null_mask, steps=20, cfg_scale=3.5, shift=3.0)
            img_t = ((sample_t.float().cpu() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy()
            batch_outputs["teacher"] = img_t
            dt_t = time.time() - t0
        else:
            dt_t = 0.0

        # 2. Our Fast EMA
        set_gen_weights(fast_ema_state)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            sample_fast = rollout_dataspace(model, z, y, mask, r=args.steps - 1, track_grad=False)
        img_fast = ((sample_fast.float().cpu() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        batch_outputs["our_fast_ema"] = img_fast

        # 3. Our Slow EMA
        set_gen_weights(slow_ema_state)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            sample_slow = rollout_dataspace(model, z, y, mask, r=args.steps - 1, track_grad=False)
        img_slow = ((sample_slow.float().cpu() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        batch_outputs["our_slow_ema"] = img_slow

        # 4. Our Mid Checkpoint
        if mid_state is not None:
            set_gen_weights(mid_state)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                sample_mid = rollout_dataspace(model, z, y, mask, r=args.steps - 1, track_grad=False)
            img_mid = ((sample_mid.float().cpu() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy()
            batch_outputs["our_mid_ckpt"] = img_mid

        # Async metrics calculation & image writes
        def process_batch(b_outputs: dict[str, np.ndarray], b_meta: list[dict], b_indices: list[int]):
            batch_teacher = b_outputs.get("teacher")
            for local_i, (p_meta, g_idx) in enumerate(zip(b_meta, b_indices, strict=True)):
                pid = p_meta["id"]
                grid_imgs = []
                grid_labels = []

                teacher_single = batch_teacher[local_i] if batch_teacher is not None else None

                for model_name in models_to_run:
                    img_single = b_outputs[model_name][local_i]
                    im_pil = Image.fromarray((img_single * 255.0).round().astype(np.uint8))
                    prompt_text = p_meta.get("prompt", "")

                    metrics = compute_image_metrics(
                        img_single,
                        teacher_single if model_name != "teacher" else None,
                        img_pil=im_pil,
                        prompt=prompt_text,
                        lpips_eval=lpips_eval,
                        clip_eval=clip_eval,
                    )
                    metrics.update({
                        "prompt_id": pid,
                        "category": p_meta.get("category", "general"),
                        "prompt": prompt_text,
                        "model": model_name,
                    })
                    metrics_by_model[model_name].append(metrics)

                    # Save single image
                    model_dir = images_dir / model_name
                    model_dir.mkdir(parents=True, exist_ok=True)
                    im_pil.save(model_dir / f"{pid}.png")

                    grid_imgs.append(img_single)
                    grid_labels.append(model_name)

                # Save comparison grid
                save_grid(grid_imgs, grid_labels, grids_dir / f"{pid}_grid.png")

        futures.append(executor.submit(process_batch, batch_outputs, batch_meta, batch_indices))
        print(f"Batch {b_idx + 1}/{total_batches} ({cur_b} prompts) rendered. Teacher: {dt_t:.1f}s", flush=True)

    # Wait for all background tasks
    executor.shutdown(wait=True)
    t_total = time.time() - t_start
    print(f"\nAll {num_prompts} prompts evaluated in {t_total:.1f}s ({t_total / 60:.2f} min).", flush=True)

    # Aggregate metrics
    summary = {}
    for model_name, records in metrics_by_model.items():
        if not records:
            continue
        summary[model_name] = {
            "count": len(records),
            "pixel_std": round(float(np.mean([r["pixel_std"] for r in records])), 4),
            "laplacian_var": round(float(np.mean([r["laplacian_var"] for r in records])), 6),
            "mean_saturation": round(float(np.mean([r["mean_saturation"] for r in records])), 4),
            "clip_black_pct": round(float(np.mean([r["clip_black_pct"] for r in records])), 2),
            "clip_white_pct": round(float(np.mean([r["clip_white_pct"] for r in records])), 2),
        }
        if model_name != "teacher" and "mae_vs_teacher" in records[0]:
            summary[model_name]["mae_vs_teacher"] = round(float(np.mean([r["mae_vs_teacher"] for r in records])), 4)
            summary[model_name]["psnr_vs_teacher"] = round(float(np.mean([r["psnr_vs_teacher"] for r in records])), 2)
            summary[model_name]["ssim_vs_teacher"] = round(float(np.mean([r["ssim_vs_teacher"] for r in records])), 4)
            if "lpips_vs_teacher" in records[0]:
                summary[model_name]["lpips_vs_teacher"] = round(float(np.mean([r["lpips_vs_teacher"] for r in records])), 4)
        if "clip_score" in records[0]:
            summary[model_name]["clip_score"] = round(float(np.mean([r["clip_score"] for r in records])), 4)

    # Save summary JSON
    results = {
        "summary": summary,
        "detail": metrics_by_model,
        "config": {
            "num_prompts": num_prompts,
            "hw": args.hw,
            "steps": args.steps,
            "seed": args.seed,
            "total_time_seconds": round(t_total, 2),
        },
    }
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Table
    md_lines = [
        "# Evaluation Benchmark Results\n",
        f"- Total Prompts: {num_prompts}",
        f"- Resolution: {args.hw[0]}x{args.hw[1]}",
        f"- Total Benchmark Time: {t_total:.1f}s (~{t_total/60:.1f} min)\n",
        "| Model | Steps | Pixel Std | Sharpness | Saturation | MAE vs Teacher | PSNR (dB) | SSIM | LPIPS | CLIP Score |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model_name, s in summary.items():
        steps_str = "20" if model_name == "teacher" else str(args.steps)
        mae = s.get("mae_vs_teacher", "-")
        psnr = s.get("psnr_vs_teacher", "-")
        ssim = s.get("ssim_vs_teacher", "-")
        lpips = s.get("lpips_vs_teacher", "-")
        clip = s.get("clip_score", "-")
        md_lines.append(
            f"| {model_name} | {steps_str} | {s['pixel_std']} | {s['laplacian_var']} | {s['mean_saturation']} | {mae} | {psnr} | {ssim} | {lpips} | {clip} |"
        )

    md_content = "\n".join(md_lines)
    (out_dir / "benchmark_table.md").write_text(md_content)
    print("\nBenchmark Summary:\n")
    print(md_content)


if __name__ == "__main__":
    main()
