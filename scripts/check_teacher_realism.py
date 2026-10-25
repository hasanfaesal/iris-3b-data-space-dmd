"""Render base Iris-3B teacher on 8 photographic test prompts and compute quality metrics.

Usage:
    python scripts/check_teacher_realism.py --checkpoint /home/dev/models/iris-3b --out runs/teacher_realism_check
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "vendor" / "iris-3b"),
    str(ROOT / "tests"),
]

TEST_PROMPTS = [
    {
        "id": "01_portrait",
        "category": "portrait",
        "prompt": "close-up portrait photo of an elderly fisherman with weathered skin and grey beard, soft natural overcast lighting",
    },
    {
        "id": "02_mug",
        "category": "indoor_object",
        "prompt": "a studio photograph of a ceramic coffee mug on a rough wooden table, warm morning window light, shallow depth of field",
    },
    {
        "id": "03_fox",
        "category": "wildlife",
        "prompt": "a high-detail wildlife photograph of a red fox standing in fresh snow, crisp focus",
    },
    {
        "id": "04_tokyo_street",
        "category": "street",
        "prompt": "a 35mm street photograph of people walking past a ramen shop in Tokyo on a rainy evening, wet asphalt reflections",
    },
    {
        "id": "05_mountains",
        "category": "landscape",
        "prompt": "a wide landscape photograph of jagged mountain peaks at sunset, golden hour light, pine trees in foreground",
    },
    {
        "id": "06_living_room",
        "category": "interior",
        "prompt": "a photograph of a minimalist mid-century modern living room with large glass windows overlooking a garden",
    },
    {
        "id": "07_pho",
        "category": "food",
        "prompt": "a food photography shot of a bowl of hot Vietnamese pho with herbs, sliced beef, steam rising, macro details",
    },
    {
        "id": "08_night_city",
        "category": "night_light",
        "prompt": "a night photograph of a busy city intersection illuminated by neon signs and car headlights, long exposure trails",
    },
]


def compute_image_metrics(img_np: np.ndarray) -> dict:
    """Compute basic diagnostic metrics on [0, 1] RGB numpy array (H, W, 3)."""
    # 1. Pixel Mean & Std
    mean = float(img_np.mean())
    std = float(img_np.std())

    # 2. Dynamic range / clipping
    clip_black = float((img_np < 0.02).mean())
    clip_white = float((img_np > 0.98).mean())

    # 3. Color saturation (HSV)
    max_c = img_np.max(axis=-1)
    min_c = img_np.min(axis=-1)
    delta = max_c - min_c
    sat = np.zeros_like(max_c)
    mask = max_c > 1e-5
    sat[mask] = delta[mask] / max_c[mask]
    mean_sat = float(sat.mean())

    # 4. Spatial sharpness via Laplacian variance (on grayscale)
    gray = 0.2989 * img_np[..., 0] + 0.5870 * img_np[..., 1] + 0.1140 * img_np[..., 2]
    # Simple discrete Laplacian kernel [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    lap = (
        np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
        - 4.0 * gray
    )
    lap_var = float(lap.var())

    return {
        "pixel_mean": round(mean, 4),
        "pixel_std": round(std, 4),
        "mean_saturation": round(mean_sat, 4),
        "laplacian_var": round(lap_var, 6),
        "clip_black_pct": round(clip_black * 100, 2),
        "clip_white_pct": round(clip_white * 100, 2),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/home/dev/models/iris-3b")
    ap.add_argument("--out", default="runs/teacher_realism_check")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--hw", type=int, nargs=2, default=[1024, 1024])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...", flush=True)
    import iris3b.text  # noqa: F401
    from iris3b.config import inference_config
    from iris3b.flow.solver import FlowDPMSolver
    from iris3b.registry import TEXT_ENCODERS
    from iris3b.sampling import load_for_inference
    from src.models.loading import load_dataspace_iris

    device = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = args.hw

    # Load text encoder
    raw, _ = load_for_inference(args.checkpoint)
    cfg = inference_config(raw, [])
    enc = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=device)

    # Load base Iris DiT
    model, _ = load_dataspace_iris(args.checkpoint)
    model.eval()

    prompts_text = [p["prompt"] for p in TEST_PROMPTS]
    print(f"Encoding {len(prompts_text)} prompts with Qwen3-VL...", flush=True)
    encoded = enc.encode(prompts_text)
    null_enc = enc.null("")

    y_all = encoded.embeddings.to(torch.bfloat16).to(device)
    mask_all = encoded.mask.to(device)
    null_emb = null_enc.embeddings.to(torch.bfloat16).to(device)
    null_mask = null_enc.mask.to(device)

    # Pad null to match y_all token length
    if null_emb.shape[1] < y_all.shape[1]:
        pad_len = y_all.shape[1] - null_emb.shape[1]
        null_emb = torch.cat(
            [null_emb, torch.zeros(1, pad_len, *null_emb.shape[2:], dtype=null_emb.dtype, device=device)],
            dim=1,
        )
        null_mask = torch.cat([null_mask, torch.zeros(1, pad_len, dtype=null_mask.dtype, device=device)], dim=1)

    all_metrics = []
    print(f"Sampling {len(TEST_PROMPTS)} images ({h}x{w}, steps={args.steps}, cfg={args.cfg})...", flush=True)

    for i, meta in enumerate(TEST_PROMPTS):
        pid = meta["id"]
        prompt = meta["prompt"]
        print(f"\n[{i+1}/{len(TEST_PROMPTS)}] Rendering: {pid}", flush=True)
        print(f"Prompt: \"{prompt}\"", flush=True)

        t0 = time.time()
        g = torch.Generator(device=device).manual_seed(args.seed + i)
        z = torch.randn(1, 3, h, w, device=device, generator=g).float()

        y = y_all[i : i + 1]
        mask = mask_all[i : i + 1]
        null = null_emb.expand(1, -1, -1, -1)
        cfg_mask = torch.cat([null_mask, mask])

        def model_fn(x, t, cond, y=y, mask=mask, cfg_mask=cfg_mask):
            m = mask if cond is y else cfg_mask
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                return model(x, t, cond, y_mask=m).x.float()

        solver = FlowDPMSolver(model_fn, num_timesteps=1000, cfg_scale=args.cfg, prediction="v")
        sample = solver.sample(z, y, null if args.cfg != 1.0 else None, steps=args.steps, order=2, shift=args.shift)
        sample = sample.clamp(-1, 1)

        # Convert to numpy uint8
        img_tensor = ((sample[0].float().cpu() + 1) / 2).clamp(0, 1)  # [3, H, W]
        img_np = img_tensor.permute(1, 2, 0).numpy()  # [H, W, 3] in [0, 1]

        # Save image
        save_path = out_dir / f"{pid}.png"
        Image.fromarray((img_np * 255.0).round().astype(np.uint8)).save(save_path)

        dt = time.time() - t0
        metrics = compute_image_metrics(img_np)
        metrics["id"] = pid
        metrics["category"] = meta["category"]
        metrics["prompt"] = prompt
        metrics["render_sec"] = round(dt, 2)
        all_metrics.append(metrics)

        print(
            f"Saved: {save_path} in {dt:.1f}s | std={metrics['pixel_std']} "
            f"lap_var={metrics['laplacian_var']} sat={metrics['mean_saturation']}",
            flush=True,
        )

    # Save summary json
    summary_path = out_dir / "metrics.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nAll renders completed. Metrics saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
