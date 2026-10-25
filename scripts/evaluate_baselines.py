#!/usr/bin/env python3
"""Run evaluation for external 4-step baselines (Pentacoxian DMD2 LoRA and Official Iris-3B Turbo).

Appends results directly into runs/benchmark_eval_200/eval_results.json and benchmark_table.md.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "vendor" / "iris-3b" / "src"),
    str(ROOT / "vendor" / "iris-3b-turbo" / "src"),
]

from iris3b.config import ModelConfig, inference_config
from iris3b.flow.schedule import shift_sigma
from iris3b.models.dit import IrisDiT
from iris3b.sampling import load_for_inference
from scripts.evaluate import (
    OptionalCLIPScore,
    OptionalLPIPS,
    compute_image_metrics,
    load_prompts,
)
from src.data.embed_cache import EmbedCache
from src.models.lora import merge_lora_file


def turbo_times(steps: int = 4, shift: float = 4.0) -> list[float]:
    grid = 1.0 - torch.arange(steps, dtype=torch.float64) / steps
    return shift_sigma(grid, shift).tolist()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="/home/dev/models/iris-3b")
    ap.add_argument(
        "--pentacoxian-adapter",
        default="/home/dev/.cache/huggingface/hub/models--pentacoxian-dev--iris-3b-turbo-lora/snapshots/fe16b3496eb440585993c92d95641c08c6b08263/r2-1000/student.safetensors",
    )
    ap.add_argument(
        "--official-model",
        default="/home/dev/.cache/huggingface/hub/models--speridlabs--iris-3b/snapshots/adc05b909fb05d4276d26aae5a52b36406b2ef26/turbo/model.safetensors",
    )
    ap.add_argument(
        "--official-config",
        default="/home/dev/.cache/huggingface/hub/models--speridlabs--iris-3b/snapshots/adc05b909fb05d4276d26aae5a52b36406b2ef26/turbo/config.yaml",
    )
    ap.add_argument("--eval-prompts", default="data/prompts_eval.jsonl")
    ap.add_argument("--eval-cache", default="/home/dev/cache_eval")
    ap.add_argument("--out", default="runs/benchmark_eval_200")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hw", type=int, nargs=2, default=[1024, 1024])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    images_dir = out_dir / "images"

    # 1. Load Prompts & Cache
    prompts_meta = load_prompts(Path(args.eval_prompts))
    num_prompts = len(prompts_meta)
    print(f"Loaded {num_prompts} prompts from {args.eval_prompts}")

    cache = EmbedCache(Path(args.eval_cache))

    def get_batch_text(indices: list[int]):
        y, mask = cache.get(indices)
        return y.to(device, dtype=torch.bfloat16), mask.to(device)

    # 2. Check existing teacher images
    teacher_img_dir = images_dir / "teacher"
    if not teacher_img_dir.exists():
        raise RuntimeError(f"Teacher images directory not found at {teacher_img_dir}")

    # 3. Setup Pentacoxian Model
    print(f"Loading Base Model + Merging Pentacoxian LoRA ({args.pentacoxian_adapter})...")
    raw_cfg, base_weights = load_for_inference(args.base_model)
    cfg_inf = inference_config(raw_cfg, [])
    with torch.device("meta"):
        penta_model = IrisDiT(cfg_inf.model)
    penta_model.load_state_dict(base_weights, strict=True, assign=True)
    del base_weights
    penta_meta = merge_lora_file(penta_model, args.pentacoxian_adapter)
    print(f"Pentacoxian merged: {penta_meta}")
    penta_model = penta_model.to(device=device, dtype=torch.bfloat16).eval()

    # 4. Setup Official Iris-3B Turbo Model
    print(f"Loading Official Iris-3B Turbo ({args.official_model})...")
    conf = OmegaConf.load(args.official_config)
    model_cfg = ModelConfig(**conf.model)
    with torch.device("meta"):
        official_model = IrisDiT(model_cfg)
    official_state = load_file(args.official_model)
    official_model.load_state_dict(official_state, strict=True, assign=True)
    del official_state
    official_model = official_model.to(device=device, dtype=torch.bfloat16).eval()

    # Evaluators
    lpips_eval = OptionalLPIPS(device)
    clip_eval = OptionalCLIPScore(device)

    models_to_run = ["pentacoxian_turbo", "official_turbo"]
    metrics_by_model: dict[str, list[dict]] = {m: [] for m in models_to_run}
    executor = ThreadPoolExecutor(max_workers=6)
    futures = []

    h, w = args.hw
    total_batches = (num_prompts + args.batch_size - 1) // args.batch_size
    penta_times = turbo_times(4, 4.0)
    official_sigmas = [1.0, 0.938, 0.833, 0.625]

    t_start = time.time()
    for b_idx in range(total_batches):
        start_idx = b_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, num_prompts)
        batch_indices = list(range(start_idx, end_idx))
        cur_b = len(batch_indices)
        batch_meta = [prompts_meta[i] for i in batch_indices]

        y, mask = get_batch_text(batch_indices)

        # Matched noise generator
        g = torch.Generator(device=device).manual_seed(args.seed + b_idx)
        z = torch.randn(cur_b, 3, h, w, device=device, generator=g)

        batch_outputs: dict[str, np.ndarray] = {}

        # 1. Pentacoxian Turbo (4 steps)
        t0 = time.time()
        x_penta = z.clone()
        for i, t_now in enumerate(penta_times):
            t_tensor = torch.full((cur_b,), t_now * 1000.0, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                v = penta_model(x_penta.to(torch.bfloat16), t_tensor, y, y_mask=mask).x.float()
            x0 = x_penta - t_now * v
            if i + 1 == len(penta_times):
                sample_penta = x0.clamp(-1, 1)
                break
            fresh = torch.randn(x_penta.shape, generator=g, device=device)
            x_penta = (1.0 - penta_times[i + 1]) * x0 + penta_times[i + 1] * fresh

        img_penta = ((sample_penta.float().cpu() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        batch_outputs["pentacoxian_turbo"] = img_penta
        dt_penta = time.time() - t0

        # 2. Official Turbo (4 steps)
        t1 = time.time()
        x_off = z.clone()
        sigma = torch.full((cur_b, 1, 1, 1), official_sigmas[0], device=device, dtype=torch.float32)
        for i in range(len(official_sigmas)):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                v_off = official_model(x_off.to(torch.bfloat16), sigma.flatten() * 1000.0, y, y_mask=mask).x.float()
            clean_off = x_off - sigma * v_off
            if i + 1 < len(official_sigmas):
                implied_noise = (x_off - (1 - sigma) * clean_off) / sigma
                sigma = torch.full_like(sigma, official_sigmas[i + 1])
                x_off = (1 - sigma) * clean_off + sigma * implied_noise
            else:
                sample_off = clean_off.clamp(-1, 1)

        img_off = ((sample_off.float().cpu() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        batch_outputs["official_turbo"] = img_off
        dt_off = time.time() - t1

        # Process Batch Async
        def process_batch(b_outputs: dict[str, np.ndarray], b_meta: list[dict]):
            for local_i, p_meta in enumerate(b_meta):
                pid = p_meta["id"]
                prompt_text = p_meta.get("prompt", "")

                # Load reference teacher image
                teacher_path = teacher_img_dir / f"{pid}.png"
                teacher_np = None
                if teacher_path.exists():
                    teacher_np = np.array(Image.open(teacher_path)).astype(np.float32) / 255.0

                for m_name in models_to_run:
                    img_single = b_outputs[m_name][local_i]
                    im_pil = Image.fromarray((img_single * 255.0).round().astype(np.uint8))

                    metrics = compute_image_metrics(
                        img_single,
                        teacher_np,
                        img_pil=im_pil,
                        prompt=prompt_text,
                        lpips_eval=lpips_eval,
                        clip_eval=clip_eval,
                    )
                    metrics.update({
                        "prompt_id": pid,
                        "category": p_meta.get("category", "general"),
                        "prompt": prompt_text,
                        "model": m_name,
                    })
                    metrics_by_model[m_name].append(metrics)

                    # Save image
                    m_dir = images_dir / m_name
                    m_dir.mkdir(parents=True, exist_ok=True)
                    im_pil.save(m_dir / f"{pid}.png")

        futures.append(executor.submit(process_batch, batch_outputs, batch_meta))
        print(f"Batch {b_idx + 1}/{total_batches} done. Penta: {dt_penta:.2f}s, Official: {dt_off:.2f}s", flush=True)

    executor.shutdown(wait=True)
    t_total = time.time() - t_start
    print(f"\nCompleted baselines evaluation in {t_total:.1f}s.")

    # Load existing results to merge
    res_path = out_dir / "eval_results.json"
    with open(res_path) as f:
        full_res = json.load(f)

    summary = full_res.setdefault("summary", {})
    detail = full_res.setdefault("detail", {})

    for m_name in models_to_run:
        records = metrics_by_model[m_name]
        detail[m_name] = records
        s = {
            "count": len(records),
            "pixel_std": round(float(np.mean([r["pixel_std"] for r in records])), 4),
            "laplacian_var": round(float(np.mean([r["laplacian_var"] for r in records])), 6),
            "mean_saturation": round(float(np.mean([r["mean_saturation"] for r in records])), 4),
            "clip_black_pct": round(float(np.mean([r["clip_black_pct"] for r in records])), 2),
            "clip_white_pct": round(float(np.mean([r["clip_white_pct"] for r in records])), 2),
            "mae_vs_teacher": round(float(np.mean([r["mae_vs_teacher"] for r in records])), 4),
            "psnr_vs_teacher": round(float(np.mean([r["psnr_vs_teacher"] for r in records])), 2),
            "ssim_vs_teacher": round(float(np.mean([r["ssim_vs_teacher"] for r in records])), 4),
        }
        if "lpips_vs_teacher" in records[0]:
            s["lpips_vs_teacher"] = round(float(np.mean([r["lpips_vs_teacher"] for r in records])), 4)
        if "clip_score" in records[0]:
            s["clip_score"] = round(float(np.mean([r["clip_score"] for r in records])), 4)
        summary[m_name] = s

    # Write merged eval_results.json
    with open(res_path, "w") as f:
        json.dump(full_res, f, indent=2)

    # Format Markdown Table
    md_lines = [
        "# Evaluation Benchmark Results\n",
        f"- Total Prompts: {num_prompts}",
        f"- Resolution: {args.hw[0]}x{args.hw[1]}",
        f"- Total Benchmark Time: {full_res.get('config', {}).get('total_time_seconds', 0) + round(t_total, 1)}s\n",
        "| Model | Steps | Pixel Std | Sharpness | Saturation | MAE vs Teacher | PSNR (dB) | SSIM | LPIPS | CLIP Score |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model_name, s in summary.items():
        steps_str = "20" if model_name == "teacher" else "4"
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
    print("\n" + md_content)


if __name__ == "__main__":
    main()
