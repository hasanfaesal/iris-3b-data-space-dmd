"""Smoke tests for scripts/infer.py."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from infer import run_inference, tensor_to_pil  # noqa: E402


def test_tensor_to_pil_bounds():
    x = torch.tensor([[[-2.0, 0.0], [1.0, 2.0]]] * 3)
    img = tensor_to_pil(x)
    assert img.mode == "RGB"
    assert img.size == (2, 2)
    assert img.getpixel((0, 0)) == (0, 0, 0)
    assert img.getpixel((0, 1)) == (255, 255, 255)


def test_tiny_inference_deterministic():
    img1 = run_inference(prompt="test", base_model="", adapter="", steps=4, seed=7, device="cpu", tiny=True)
    img2 = run_inference(prompt="test", base_model="", adapter="", steps=4, seed=7, device="cpu", tiny=True)
    assert img1.size == (32, 32)
    assert img1.tobytes() == img2.tobytes()
