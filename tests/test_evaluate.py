"""Test for scripts/evaluate.py metrics calculation, grid output, and data parsing."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from scripts.evaluate import compute_image_metrics, save_grid, load_prompts


def test_compute_image_metrics_basic():
    # Synthetic flat image
    img = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    metrics = compute_image_metrics(img)
    assert metrics["pixel_mean"] == 0.5
    assert metrics["pixel_std"] == 0.0
    assert metrics["mean_saturation"] == 0.0
    assert metrics["clip_black_pct"] == 0.0
    assert metrics["clip_white_pct"] == 0.0


def test_compute_image_metrics_with_teacher():
    img1 = np.ones((64, 64, 3), dtype=np.float32) * 0.4
    teacher = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    metrics = compute_image_metrics(img1, teacher)
    assert metrics["mae_vs_teacher"] == 0.1
    assert "psnr_vs_teacher" in metrics
    assert metrics["psnr_vs_teacher"] > 0
    assert "ssim_vs_teacher" in metrics
    assert 0.0 <= metrics["ssim_vs_teacher"] <= 1.0


def test_save_grid(tmp_path):
    img1 = np.zeros((32, 32, 3), dtype=np.float32)
    img2 = np.ones((32, 32, 3), dtype=np.float32)
    out_path = tmp_path / "grid.png"
    save_grid([img1, img2], ["M1", "M2"], out_path)
    assert out_path.exists()
    im = Image.open(out_path)
    assert im.size[0] == 64  # 32 + 32
    assert im.size[1] == 32 + 32  # 32 + 32 header


def test_load_prompts(tmp_path):
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(
        json.dumps({"prompt": "A cat", "category": "Animals"}) + "\n" +
        json.dumps({"prompt": "A sunset", "category": "Landscape"}) + "\n"
    )
    loaded = load_prompts(prompts_file, limit=1)
    assert len(loaded) == 1
    assert loaded[0]["prompt"] == "A cat"
    assert loaded[0]["id"] == "0000"


def test_evaluate_end_to_end_synthetic(tmp_path, monkeypatch):
    """End-to-end integration test running main() with tiny model, mock checkpoints, and cached embeds."""
    import sys
    from src.models.lora import inject_lora
    from tiny import dataspace_iris, tiny_cond
    from scripts import evaluate

    # 1. Create prompts file
    prompts_file = tmp_path / "prompts_eval.jsonl"
    prompts_file.write_text(
        json.dumps({"prompt": "A cat on the grass", "category": "Animals"}) + "\n" +
        json.dumps({"prompt": "A sunset on a beach", "category": "Landscape"}) + "\n"
    )

    # 2. Create mock embed cache with null.pt and index.json
    cache_dir = tmp_path / "cache_eval"
    cache_dir.mkdir(parents=True, exist_ok=True)
    null_y, null_mask = tiny_cond(1)
    torch.save({"emb": null_y, "mask": null_mask}, cache_dir / "null.pt")
    (cache_dir / "index.json").write_text(json.dumps({"n": 2, "max_tokens": 16}))

    # Mock EmbedCache.get to return deterministic tiny embeddings
    def mock_get(self, indices):
        y, mask = tiny_cond(len(indices))
        return y, mask

    monkeypatch.setattr(evaluate.EmbedCache, "get", mock_get)

    # 3. Create tiny base model and inject LoRA to prepare checkpoint
    base_model = dataspace_iris()
    inject_lora(base_model, "gen", rank=8, alpha=8.0)

    # Build checkpoint in state_dict_slim format (matching Trainer)
    named = dict(base_model.named_parameters())
    gen_params = {n: p.detach().cpu() for n, p in named.items() if ".adapters.gen." in n or n.startswith(("x0_embedder.", "step_embeddings."))}
    ckpt_path = tmp_path / "last.pt"
    torch.save({
        "gen": gen_params,
        "ema": gen_params,
        "ema_slow": gen_params,
        "gen_updates": 100,
    }, ckpt_path)

    # Mid checkpoint
    mid_ckpt_path = tmp_path / "mid.pt"
    torch.save({
        "gen": gen_params,
        "ema": gen_params,
    }, mid_ckpt_path)

    # Mock load_dataspace_iris to return a fresh tiny model with rank=8 LoRA injected
    def mock_load_dataspace_iris(model_path, **kwargs):
        m = dataspace_iris()
        return m, None

    monkeypatch.setattr(evaluate, "load_dataspace_iris", mock_load_dataspace_iris)

    # Mock inject_lora so it matches the test model rank (8)
    real_inject_lora = evaluate.inject_lora
    def mock_inject_lora(model, name, rank=64, alpha=64.0):
        return real_inject_lora(model, name, rank=8, alpha=8.0)

    monkeypatch.setattr(evaluate, "inject_lora", mock_inject_lora)

    out_dir = tmp_path / "eval_out"

    test_args = [
        "evaluate.py",
        "--base-model", str(tmp_path / "mock_base"),
        "--checkpoint", str(ckpt_path),
        "--mid-checkpoint", str(mid_ckpt_path),
        "--eval-prompts", str(prompts_file),
        "--eval-cache", str(cache_dir),
        "--out", str(out_dir),
        "--batch-size", "2",
        "--steps", "2",
        "--hw", "16", "16",
        "--seed", "42",
        "--skip-teacher",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Run main() end-to-end
    evaluate.main()

    # Verify generated outputs
    results_file = out_dir / "eval_results.json"
    assert results_file.exists()
    results = json.loads(results_file.read_text())
    assert "summary" in results
    assert "our_fast_ema" in results["summary"]
    assert "our_slow_ema" in results["summary"]
    assert "our_mid_ckpt" in results["summary"]

    # Verify markdown table
    table_file = out_dir / "benchmark_table.md"
    assert table_file.exists()
    table_text = table_file.read_text()
    assert "| our_fast_ema |" in table_text
    assert "{'mae_vs_teacher'" not in table_text  # confirms {s} bug is not present

    # Verify images and grids
    assert (out_dir / "images" / "our_fast_ema" / "0000.png").exists()
    assert (out_dir / "grids" / "0000_grid.png").exists()
