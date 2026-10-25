#!/usr/bin/env python3
"""Background uploader for all artifacts to private Hugging Face repo."""

import os
from pathlib import Path
import time
from huggingface_hub import HfApi

REPO_ID = "hasanfaesal/iris-3b-dataspace-dmd-turbo"
ROOT = Path("/home/dev/h100-2")

api = HfApi()

upload_plan = [
    # 1. Core Model & Evaluation Artifacts
    (ROOT / "exported/iris-3b-dataspace-turbo-lora.safetensors", "iris-3b-dataspace-turbo-lora.safetensors"),
    (ROOT / "exported/eval_images.zip", "eval/eval_images.zip"),
    (ROOT / "exported/eval_results.json", "eval/eval_results.json"),
    (ROOT / "exported/benchmark_table.md", "eval/benchmark_table.md"),
    (ROOT / "exported/cache_eval_200.zip", "eval/cache_eval_200.zip"),
    
    # 2. Training Logs & Previews
    (ROOT / "runs/main/metrics.jsonl", "training/metrics.jsonl"),
    (ROOT / "runs/main/status.json", "training/status.json"),
    (ROOT / "exported/training_previews.zip", "training/training_previews.zip"),
    
    # 3. Model Checkpoints
    (ROOT / "runs/main/checkpoints/last_good.pt", "checkpoints/last_good.pt"),
    (ROOT / "runs/main/checkpoints/last.pt", "checkpoints/last.pt"),
]

# Add milestone checkpoints
milestones = sorted((ROOT / "runs/main/checkpoints").glob("gen*.pt"))
for m in milestones:
    upload_plan.append((m, f"checkpoints/{m.name}"))

# 4. Generate README.md Model Card
bench_table = (ROOT / "exported/benchmark_table.md").read_text() if (ROOT / "exported/benchmark_table.md").exists() else ""
readme_content = f"""---
license: apache-2.0
base_model: speridlabs/iris-3b
tags:
- text-to-image
- pixel-space
- diffusion
- distillation
- lora
- dmd2
- data-space
---

# Iris-3B Data-Space DMD Turbo (4-Step LoRA)

This repository contains the complete release artifacts, training logs, evaluation benchmarks, and raw checkpoints for the **Iris-3B Data-Space DMD Turbo** 4-step generator LoRA.

{bench_table}

## Architecture Details
- **Base Model**: Sperid Labs `iris-3b` (3B parameter pixel-space diffusion transformer)
- **Distillation Method**: Data-space Distribution Matching Distillation (DSDMD)
- **Adapter**: Rank 128 / Alpha 128 LoRA with integrated data-space prediction branch (`x0_embedder` and `step_embeddings`)
- **Sampling Steps**: 4 steps (Euler/Flow with shift 4.0)

## Repository Structure
- `iris-3b-dataspace-turbo-lora.safetensors`: Primary 4-step LoRA generator adapter.
- `eval/`: Complete 200-prompt benchmark metrics and all 1,200 generated images across all 6 models.
- `training/`: Training step metrics (`metrics.jsonl`) and full image preview history (`training_previews.zip`).
- `checkpoints/`: Full training checkpoints (`last_good.pt`, `last.pt`) and 9 step milestones (`gen*.pt`).
"""
readme_path = ROOT / "exported/README.md"
readme_path.write_text(readme_content)
upload_plan.append((readme_path, "README.md"))

print(f"Starting direct upload of {len(upload_plan)} artifacts to {REPO_ID}...")
total_start = time.time()

for idx, (local_path, path_in_repo) in enumerate(upload_plan, 1):
    if not local_path.exists():
        print(f"[{idx}/{len(upload_plan)}] SKIP (missing): {local_path}")
        continue
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"[{idx}/{len(upload_plan)}] Uploading {path_in_repo} ({size_mb:.2f} MB)...", flush=True)
    t0 = time.time()
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type="model",
        )
        dt = time.time() - t0
        speed_mbps = (size_mb / dt) if dt > 0 else 0
        print(f"  -> Done in {dt:.1f}s ({speed_mbps:.1f} MB/s)", flush=True)
    except Exception as e:
        print(f"  -> ERROR uploading {path_in_repo}: {e}", flush=True)

total_dt = time.time() - total_start
print(f"\nAll uploads completed in {total_dt:.1f}s!")
