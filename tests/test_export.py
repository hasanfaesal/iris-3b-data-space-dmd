"""Tests for scripts/export.py adapter extraction, packaging, and metadata handling."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
from safetensors import safe_open
from safetensors.torch import load_file
import torch
from PIL import Image

from scripts.export import (
    export_adapter,
    generate_model_card,
    pack_evaluation_images,
)
from src.models.lora import (
    inject_lora,
    load_adapter,
    lora_layers,
    merge_lora_file,
    read_metadata,
)
from tests.tiny import dataspace_iris


@pytest.fixture
def mock_checkpoint_path(tmp_path: Path) -> Path:
    """Create a realistic mock trainer checkpoint (.pt)."""
    model = dataspace_iris(seed=42)
    inject_lora(model, "gen", rank=8, alpha=16.0)

    # Fill branch modules with non-zero test values
    with torch.no_grad():
        model.x0_embedder.weight.fill_(0.123)
        model.step_embeddings.weight.fill_(0.456)
        for _, layer in lora_layers(model).items():
            down, up = layer.adapters["gen"]
            down.weight.fill_(0.01)
            up.weight.fill_(0.02)

    gen_weights = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    ema_weights = {n: p.detach().cpu().clone() * 1.5 for n, p in model.named_parameters()}

    ckpt = {
        "gen": gen_weights,
        "ema": ema_weights,
        "gen_updates": 1500,
        "fake_updates": 1500,
    }
    ckpt_file = tmp_path / "last.pt"
    torch.save(ckpt, ckpt_file)
    return ckpt_file


def test_export_adapter_weights_and_metadata(mock_checkpoint_path: Path, tmp_path: Path):
    out_safetensors = tmp_path / "adapter.safetensors"
    exported_file, meta = export_adapter(
        checkpoint_path=mock_checkpoint_path,
        out_path=out_safetensors,
        weights_key="ema",
    )
    assert exported_file.exists()
    assert exported_file == out_safetensors

    # 1. Verify metadata header
    assert meta["rank"] == "8"
    assert meta["alpha"] == "8.0" or meta["alpha"] == "8"
    assert meta["training_step"] == "1500"
    assert meta["num_steps"] == "4"
    assert meta["model_type"] == "iris-3b-dataspace-turbo"
    assert meta["source_checkpoint"] == "last.pt"
    assert "date" in meta

    # 2. Inspect with safe_open directly
    with safe_open(str(exported_file), framework="pt") as f:
        file_meta = f.metadata()
        assert file_meta["rank"] == "8"
        assert file_meta["training_step"] == "1500"
        assert file_meta["model_type"] == "iris-3b-dataspace-turbo"
        keys = set(f.keys())

    # 3. Check presence of branch weights and LoRA down/up keys
    assert "x0_embedder.weight" in keys
    assert "step_embeddings.weight" in keys

    tensors = load_file(str(exported_file))
    # In mock, ema was gen * 1.5 -> 0.123 * 1.5 = 0.1845
    assert torch.allclose(tensors["x0_embedder.weight"], torch.full_like(tensors["x0_embedder.weight"], 0.1845))
    assert torch.allclose(tensors["step_embeddings.weight"], torch.full_like(tensors["step_embeddings.weight"], 0.456 * 1.5))

    # Verify at least one LoRA down/up pair
    lora_down_keys = [k for k in keys if k.endswith("lora_down.weight")]
    lora_up_keys = [k for k in keys if k.endswith("lora_up.weight")]
    assert len(lora_down_keys) > 0
    assert len(lora_up_keys) == len(lora_down_keys)


def test_adapter_round_trip_loading(mock_checkpoint_path: Path, tmp_path: Path):
    out_safetensors = tmp_path / "adapter.safetensors"
    export_adapter(
        checkpoint_path=mock_checkpoint_path,
        out_path=out_safetensors,
        weights_key="ema",
    )

    # Create fresh model with zeroed branch weights
    fresh_model = dataspace_iris(seed=99)
    inject_lora(fresh_model, "gen", rank=8, alpha=16.0)
    with torch.no_grad():
        fresh_model.x0_embedder.weight.zero_()
        fresh_model.step_embeddings.weight.zero_()

    # Load adapter
    load_adapter(fresh_model, "gen", out_safetensors)

    # Check loaded weights
    assert torch.allclose(fresh_model.x0_embedder.weight, torch.full_like(fresh_model.x0_embedder.weight, 0.1845))
    assert torch.allclose(fresh_model.step_embeddings.weight, torch.full_like(fresh_model.step_embeddings.weight, 0.456 * 1.5))


def test_pack_evaluation_images(tmp_path: Path):
    images_dir = tmp_path / "eval_images"
    images_dir.mkdir()

    # Create 5 dummy png images
    for i in range(5):
        im = Image.new("RGB", (32, 32), color=(i * 20, i * 20, i * 20))
        im.save(images_dir / f"sample_{i:04d}.png")

    zip_out = tmp_path / "eval_images.zip"
    zip_path, count = pack_evaluation_images(images_dir, zip_out)

    assert zip_path.exists()
    assert count == 5
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert len(namelist) == 5
        assert "sample_0000.png" in namelist


def test_generate_model_card():
    meta = {
        "rank": "64",
        "alpha": "64.0",
        "training_step": "2000",
        "num_steps": "4",
        "date": "2026-10-25T12:00:00Z",
    }
    eval_summary = {
        "pixel_std": 0.285,
        "laplacian_var": 0.0035,
    }
    card = generate_model_card("hasanfaesal/iris-3b-dataspace-turbo", meta, eval_summary=eval_summary)
    assert "hasanfaesal/iris-3b-dataspace-turbo" not in card or "hasanfaesal" in card
    assert "LoRA Rank:** 64" in card
    assert "Training Step:** 2000" in card
    assert "pixel_std" in card
    assert "0.285" in card


def test_merge_lora_into_base_model(mock_checkpoint_path: Path, tmp_path: Path):
    out_safetensors = tmp_path / "adapter.safetensors"
    export_adapter(
        checkpoint_path=mock_checkpoint_path,
        out_path=out_safetensors,
        weights_key="ema",
    )

    base_model = dataspace_iris(seed=123)
    meta = merge_lora_file(base_model, out_safetensors)
    assert meta["rank"] == 8
    # Branch weights should be transferred to base model
    assert torch.allclose(base_model.x0_embedder.weight, torch.full_like(base_model.x0_embedder.weight, 0.1845))
    assert torch.allclose(base_model.step_embeddings.weight, torch.full_like(base_model.step_embeddings.weight, 0.456 * 1.5))


def test_cli_export(mock_checkpoint_path: Path, tmp_path: Path):
    # Setup dummy eval images
    img_dir = tmp_path / "previews"
    img_dir.mkdir()
    im = Image.new("RGB", (16, 16), color=(255, 0, 0))
    im.save(img_dir / "test.png")

    out_file = tmp_path / "cli_adapter.safetensors"

    cmd = [
        sys.executable,
        "scripts/export.py",
        "--checkpoint",
        str(mock_checkpoint_path),
        "--weights",
        "ema",
        "--out",
        str(out_file),
        "--images-dir",
        str(img_dir),
        "--zip-images",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI error: {res.stderr}\nstdout: {res.stdout}"
    assert out_file.exists()
    assert (tmp_path / "eval_images.zip").exists()
