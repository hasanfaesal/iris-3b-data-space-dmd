#!/usr/bin/env python3
"""Export data-space generator adapter weights, package evaluation images, and publish to Hugging Face Hub.

Extracts generator LoRA adapter matrices and data-space branch weights (x0_embedder.weight,
step_embeddings.weight) from training checkpoints, writes standard safetensors with metadata,
optionally merges weights into base models, compresses/uploads evaluation images, and pushes to HF.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import zipfile

import torch

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.lora import (
    extract_adapter_from_checkpoint,
    merge_lora_file,
    read_metadata,
    save_adapter_file,
)
from src.utils.checkpoint import load_checkpoint


def export_adapter(
    checkpoint_path: str | Path,
    out_path: str | Path,
    weights_key: str = "ema",
    rank: int | None = None,
    alpha: float | None = None,
    num_steps: int | None = None,
    extra_metadata: dict | None = None,
) -> tuple[Path, dict[str, str]]:
    """Extract and save standalone data-space adapter safetensors file."""
    ckpt_path = Path(checkpoint_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tensors, metadata = extract_adapter_from_checkpoint(
        checkpoint=ckpt_path,
        weights_key=weights_key,
        rank=rank,
        alpha=alpha,
        source_name=ckpt_path.name,
    )

    if num_steps is not None:
        metadata["num_steps"] = str(num_steps)
    if extra_metadata:
        for k, v in extra_metadata.items():
            metadata[k] = str(v)

    save_adapter_file(tensors, metadata, out_path)
    return out_path, metadata


def merge_and_export(
    base_model_path: str | Path,
    adapter_path: str | Path,
    out_path: str | Path,
) -> Path:
    """Fold adapter weights into base model and export full merged safetensors."""
    from src.models.loading import load_dataspace_iris
    from safetensors.torch import save_file

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading base DataSpaceIris model from {base_model_path}...")
    model, _ = load_dataspace_iris(str(base_model_path))
    model.eval()

    print(f"Merging adapter from {adapter_path} into base model...")
    meta = merge_lora_file(model, adapter_path)
    print(f"Merged LoRA (rank={meta.get('rank')}, alpha={meta.get('alpha')}).")

    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(out_path), metadata={"merged": "true", **{k: str(v) for k, v in meta.items()}})
    return out_path


def pack_evaluation_images(
    images_dir: str | Path,
    zip_out_path: str | Path,
) -> tuple[Path, int]:
    """Compress all images in images_dir into a zip file. Returns (zip_path, image_count)."""
    img_dir = Path(images_dir)
    zip_path = Path(zip_out_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        raise FileNotFoundError(f"Evaluation images directory not found: {img_dir}")

    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = [p for p in img_dir.rglob("*") if p.suffix.lower() in valid_exts and p.is_file()]

    if not image_files:
        raise ValueError(f"No image files found in {img_dir}")

    print(f"Packing {len(image_files)} evaluation images into {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in image_files:
            rel = f.relative_to(img_dir)
            zf.write(f, arcname=str(rel))

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Created {zip_path} ({size_mb:.2f} MB, {len(image_files)} files).")
    return zip_path, len(image_files)


def generate_model_card(
    repo_id: str,
    metadata: dict[str, str],
    eval_summary: dict | None = None,
) -> str:
    """Generate Markdown model card for Hugging Face Hub."""
    rank = metadata.get("rank", "64")
    alpha = metadata.get("alpha", "64.0")
    step = metadata.get("training_step", "unknown")
    steps = metadata.get("num_steps", "4")
    date = metadata.get("date", datetime.now(timezone.utc).isoformat())

    eval_table = ""
    if eval_summary:
        eval_table = "\n### Benchmark Evaluation Metrics\n\n| Metric | Value |\n|---|---|\n"
        for k, v in eval_summary.items():
            eval_table += f"| {k} | {v} |\n"

    return f"""---
license: apache-2.0
tags:
- text-to-image
- diffusion
- flow-matching
- iris-3b
- dataspace
- fast-generation
base_model: pentaCoxian/iris-3b-turbo
---

# Iris-3B Data-Space Turbo Adapter

Fast {steps}-step data-space distillation adapter for Iris-3B.

## Model Details
- **Base Model:** Iris-3B / Iris-3B-Turbo
- **Adapter Type:** LoRA + Data-Space iteration branch (x0_embedder, step_embeddings)
- **LoRA Rank:** {rank}
- **LoRA Alpha:** {alpha}
- **Rollout Steps:** {steps} steps (r = {int(steps) - 1 if steps.isdigit() else '3'})
- **Training Step:** {step}
- **Export Date:** {date}
{eval_table}
## Usage

```python
import torch
from src.models.loading import load_dataspace_iris
from src.models.lora import inject_lora, load_adapter
from src.engine.rollout import rollout_dataspace

# 1. Load base model & inject adapter slot
model, _ = load_dataspace_iris("path/to/iris-3b")
inject_lora(model, "gen", rank={rank}, alpha={alpha})

# 2. Load exported weights
load_adapter(model, "gen", "iris3b_dataspace_turbo.safetensors")
model.eval().cuda()

# 3. Generate in {steps} steps
# rollout_dataspace(model, z, y, mask, r={int(steps) - 1 if steps.isdigit() else '3'})
```
"""


def upload_to_huggingface(
    repo_id: str,
    files_to_upload: list[tuple[Path, str]],
    folders_to_upload: list[tuple[Path, str]] | None = None,
    token: str | None = None,
    private: bool = False,
    model_card_content: str | None = None,
) -> None:
    """Upload files and folders to Hugging Face Hub using HfApi."""
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError:
        raise ImportError("huggingface_hub is required for HF upload. Install with: pip install huggingface_hub")

    hf_token = token or os.environ.get("HF_TOKEN") or get_token()
    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. Set HF_TOKEN environment variable or run 'huggingface-cli login'."
        )

    api = HfApi(token=hf_token)
    print(f"Creating / verifying Hugging Face repository '{repo_id}' (private={private})...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

    if model_card_content:
        print("Uploading README.md (model card)...")
        api.upload_file(
            path_or_fileobj=model_card_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )

    for local_path, path_in_repo in files_to_upload:
        if not local_path.exists():
            print(f"Warning: skipping missing file {local_path}")
            continue
        print(f"Uploading {local_path.name} -> {repo_id}:{path_in_repo}...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
        )

    if folders_to_upload:
        for local_dir, path_in_repo in folders_to_upload:
            if not local_dir.exists():
                print(f"Warning: skipping missing directory {local_dir}")
                continue
            print(f"Uploading folder {local_dir} -> {repo_id}:{path_in_repo}...")
            api.upload_folder(
                folder_path=str(local_dir),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
            )

    print(f"Successfully published artifacts to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/last.pt",
        help="Path to training checkpoint (.pt)",
    )
    parser.add_argument(
        "--weights",
        default="ema",
        choices=["ema", "gen", "ema_slow"],
        help="Which weights to extract (default: ema)",
    )
    parser.add_argument(
        "--out",
        default="exports/iris3b_dataspace_turbo.safetensors",
        help="Destination path for exported .safetensors",
    )
    parser.add_argument("--rank", type=int, default=None, help="LoRA rank override")
    parser.add_argument("--alpha", type=float, default=None, help="LoRA alpha override")
    parser.add_argument("--steps", type=int, default=None, help="Rollout steps override (default: 4)")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA into base model and export full base safetensors",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Path to base Iris-3B checkpoint (required if --merge is set)",
    )
    parser.add_argument(
        "--images-dir",
        "--eval-dir",
        dest="images_dir",
        default=None,
        help="Path to evaluation images directory to package / upload",
    )
    parser.add_argument(
        "--zip-images",
        "--pack-images",
        dest="zip_images",
        action="store_true",
        help="Pack evaluation images into a .zip archive before upload",
    )
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="Hugging Face repository ID (e.g. hasanfaesal/iris-3b-dataspace-turbo)",
    )
    parser.add_argument("--hf-token", default=None, help="Hugging Face authentication token")
    parser.add_argument("--private", action="store_true", help="Make Hugging Face repo private")
    parser.add_argument(
        "--create-model-card",
        action="store_true",
        help="Generate and upload a README.md model card to HF",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)

    if not ckpt_path.exists():
        print(f"Error: checkpoint file does not exist: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"--- 1. Exporting Adapter ({args.weights}) from {ckpt_path} ---")
    adapter_file, meta = export_adapter(
        checkpoint_path=ckpt_path,
        out_path=out_path,
        weights_key=args.weights,
        rank=args.rank,
        alpha=args.alpha,
        num_steps=args.steps,
    )
    print(f"Saved adapter to {adapter_file}")
    print(f"Metadata: rank={meta['rank']}, alpha={meta['alpha']}, step={meta['training_step']}, num_steps={meta['num_steps']}")

    final_model_file = adapter_file
    if args.merge:
        if not args.base_model:
            print("Error: --base-model is required when --merge is set.", file=sys.stderr)
            sys.exit(1)
        merged_out = out_path.with_name(out_path.stem + "_merged" + out_path.suffix)
        print(f"\n--- 2. Merging LoRA into base model from {args.base_model} ---")
        final_model_file = merge_and_export(
            base_model_path=args.base_model,
            adapter_path=adapter_file,
            out_path=merged_out,
        )
        print(f"Saved merged model to {final_model_file}")

    files_to_upload: list[tuple[Path, str]] = [(final_model_file, final_model_file.name)]
    folders_to_upload: list[tuple[Path, str]] = []

    # Also upload unmerged adapter if we merged
    if args.merge and adapter_file != final_model_file:
        files_to_upload.append((adapter_file, adapter_file.name))

    # Check for evaluation images
    zip_path = None
    if args.images_dir:
        img_dir = Path(args.images_dir)
        if img_dir.exists():
            print(f"\n--- 3. Packaging Evaluation Images from {img_dir} ---")
            if args.zip_images:
                zip_path = out_path.parent / "eval_images.zip"
                zip_path, img_count = pack_evaluation_images(img_dir, zip_path)
                files_to_upload.append((zip_path, "eval_images.zip"))
            else:
                folders_to_upload.append((img_dir, "eval_images"))

            # Also check if eval_results.json or metrics.jsonl exist
            for m_candidate in [img_dir / "eval_results.json", img_dir.parent / "eval_results.json", ckpt_path.parent / "metrics.jsonl"]:
                if m_candidate.exists():
                    files_to_upload.append((m_candidate, m_candidate.name))
        else:
            print(f"Warning: images directory does not exist: {img_dir}")

    # Upload to Hugging Face
    if args.hf_repo:
        print(f"\n--- 4. Publishing to Hugging Face ({args.hf_repo}) ---")
        model_card = None
        if args.create_model_card:
            eval_summary = None
            for summary_path in [
                Path(args.images_dir) / "eval_results.json" if args.images_dir else None,
                out_path.parent / "eval_results.json",
                Path("runs/benchmark_eval_200/eval_results.json"),
                Path("runs/benchmark_eval/eval_results.json"),
            ]:
                if summary_path and summary_path.exists():
                    try:
                        with open(summary_path) as f:
                            data = json.load(f)
                            eval_summary = data.get("summary")
                    except Exception:
                        pass
                    break
            model_card = generate_model_card(args.hf_repo, meta, eval_summary=eval_summary)

        upload_to_huggingface(
            repo_id=args.hf_repo,
            files_to_upload=files_to_upload,
            folders_to_upload=folders_to_upload,
            token=args.hf_token,
            private=args.private,
            model_card_content=model_card,
        )

    print("\nExport process completed successfully.")


if __name__ == "__main__":
    main()
