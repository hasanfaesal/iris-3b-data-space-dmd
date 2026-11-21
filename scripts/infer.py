#!/usr/bin/env python3
"""Generate an image in 4 steps using the Iris-3B Data-Space DMD Turbo LoRA.

Examples:
    python scripts/infer.py --prompt "an oil painting of an owl in the style of Hokusai" --out owl.png
    python scripts/infer.py --tiny --out /tmp/smoke.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "vendor" / "iris-3b" / "src"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.engine.rollout import rollout_dataspace  # noqa: E402
from src.models.loading import load_dataspace_iris  # noqa: E402
from src.models.lora import inject_lora, load_adapter, read_metadata  # noqa: E402

DEFAULT_HF_REPO = "hasanfaesal/iris-3b-dataspace-dmd-turbo"
DEFAULT_HF_FILENAME = "iris-3b-dataspace-turbo-lora.safetensors"


def resolve_adapter_path(adapter: str, filename: str = DEFAULT_HF_FILENAME) -> Path:
    """Return a local safetensors path, downloading from Hugging Face Hub if needed."""
    local = Path(adapter)
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=adapter, filename=filename))


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Convert a single [3, H, W] tensor in [-1, 1] to a PIL RGB image."""
    arr = ((x.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy(), mode="RGB")


@torch.no_grad()
def run_inference(
    prompt: str,
    base_model: str,
    adapter: str,
    steps: int = 4,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    device: str | None = None,
    tiny: bool = False,
) -> Image.Image:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gen = torch.Generator(device=dev).manual_seed(seed)

    if tiny:
        if str(ROOT / "tests") not in sys.path:
            sys.path.insert(0, str(ROOT / "tests"))
        from tiny import dataspace_iris, tiny_cond

        torch.manual_seed(seed)
        model = dataspace_iris(seed=seed).to(dev).eval()
        inject_lora(model, "gen", rank=4, alpha=4.0)
        y, y_mask = tiny_cond(batch=1, device=str(dev))
        h, w = 32, 32
        dtype = torch.float32
    else:
        import iris3b.text  # noqa: F401
        from iris3b.config import inference_config
        from iris3b.registry import TEXT_ENCODERS
        from iris3b.sampling import load_for_inference

        raw, weights = load_for_inference(base_model)
        cfg = inference_config(raw, [])

        enc = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=dev)
        encoded = enc.encode([prompt])
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        y = encoded.embeddings.to(device=dev, dtype=dtype)
        y_mask = encoded.mask.to(device=dev)
        del enc
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        adapter_path = resolve_adapter_path(adapter)
        meta = read_metadata(adapter_path)
        rank = int(meta.get("rank", 128))
        alpha = float(meta.get("alpha", float(rank)))

        model, _ = load_dataspace_iris(base_model, device=dev, weights=weights, raw=raw)
        inject_lora(model, "gen", rank=rank, alpha=alpha)
        load_adapter(model, "gen", adapter_path)
        model.eval()
        h, w = height, width

    z = torch.randn((1, 3, h, w), generator=gen, device=dev, dtype=dtype)
    sample = rollout_dataspace(model, z, y, y_mask, r=steps - 1, track_grad=False)
    return tensor_to_pil(sample[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="an owl", help="Text prompt to generate")
    ap.add_argument(
        "--base-model",
        default=os.environ.get("IRIS_MODEL_DIR", "models/iris-3b"),
        help="Path to base Iris-3B directory (config.yaml + model.safetensors)",
    )
    ap.add_argument(
        "--adapter",
        default=DEFAULT_HF_REPO,
        help="Local adapter .safetensors path or Hugging Face repo ID",
    )
    ap.add_argument("--steps", type=int, default=4, help="Number of data-space refinement steps (1-4)")
    ap.add_argument("--hw", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W"), help="Output height and width")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out", default="output.png", help="Output PNG path")
    ap.add_argument("--tiny", action="store_true", help="CPU smoke run using the synthetic tiny model")
    args = ap.parse_args()

    img = run_inference(
        prompt=args.prompt,
        base_model=args.base_model,
        adapter=args.adapter,
        steps=args.steps,
        height=args.hw[0],
        width=args.hw[1],
        seed=args.seed,
        tiny=args.tiny,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved {img.size[0]}x{img.size[1]} image to {out_path}")


if __name__ == "__main__":
    main()
