"""Cache Qwen3-VL text embeddings offline (PLAN.md "Qwen: pre-encode"), before training.

    python scripts/cache_embeddings.py --out /home/dev/cache            # H100, after model download
    python scripts/cache_embeddings.py --out /home/dev/cache --verify 10

Encodes data/prompts_train.jsonl row-order into <out>/row_XXXXXX.pt — one file per prompt,
unpadded [T,12,2560] bf16 (T = true token count) — plus <out>/null.pt and <out>/index.json.
The 200 eval prompts go to <out>_eval/ the same way. One file per prompt so training reads
~4 MB randomly per row instead of a multi-GB shard; existing row files are kept, so an
interrupted run resumes where it stopped. The Iris-3B DiT is never loaded: only
cfg.text_encoder is read from the checkpoint.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prompts(path: Path) -> list[str]:
    rows = [json.loads(l)["prompt"] for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no prompts in {path}")
    return rows


@torch.no_grad()
def cache_prompts(prompts: list[str], out: Path, enc, batch: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for b in range(0, len(prompts), batch):
        idx = range(b, min(b + batch, len(prompts)))
        if all((out / f"row_{i:06d}.pt").exists() for i in idx):
            continue
        e = enc.encode(prompts[b : b + batch])
        for j, i in enumerate(idx):
            n = int(e.mask[j].sum())
            p = out / f"row_{i:06d}.pt"
            tmp = p.with_suffix(".tmp")  # atomic: a crash never leaves a torn row file
            torch.save(e.embeddings[j, :n].to(torch.bfloat16).cpu(), tmp)
            os.replace(tmp, p)
        print(f"{out}: {idx[-1] + 1}/{len(prompts)}", flush=True)


@torch.no_grad()
def verify(prompts: list[str], out: Path, enc, n: int, batch: int) -> None:
    """PLAN.md check: re-encode the ORIGINAL batch containing each sampled row and compare
    bit-exact against the cache. Batch composition is kept because cuBLAS kernel choice
    depends on it — re-encoding one row alone can differ in bits with nothing wrong."""
    import random

    rng = random.Random(0)
    for i in rng.sample(range(len(prompts)), n):
        b = (i // batch) * batch
        e = enc.encode(prompts[b : b + batch])
        live = e.embeddings[i - b, : e.mask[i - b].sum()].to(torch.bfloat16).cpu()
        cached = torch.load(out / f"row_{i:06d}.pt", weights_only=True)
        if live.shape != cached.shape or not torch.equal(live, cached):
            raise SystemExit(f"verify FAILED at row {i}")
    print(f"verified {n}/{len(prompts)} rows bit-exact", flush=True)


def write_index(prompts_path: Path, out: Path) -> None:
    index = {
        "prompts_file": str(prompts_path),
        "sha256": sha256(prompts_path),
        "n": len(load_prompts(prompts_path)),
    }
    tmp = (out / "index.json").with_suffix(".tmp")
    tmp.write_text(json.dumps(index))
    os.replace(tmp, out / "index.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts_train.jsonl")
    ap.add_argument("--eval-prompts", default="data/prompts_eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="/home/dev/models/iris-3b")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--verify", type=int, default=0, help="re-encode N random rows and compare")
    args = ap.parse_args()
    out = Path(args.out)

    import iris3b.text  # noqa: F401  (registers encoders)
    from iris3b.config import inference_config
    from iris3b.registry import TEXT_ENCODERS
    from iris3b.sampling import load_for_inference

    raw, weights = load_for_inference(args.checkpoint)
    del weights  # the DiT is never built; only cfg.text_encoder is needed
    cfg = inference_config(raw, [])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=device)

    eval_out = out.parent / (out.name + "_eval")
    for prompts_path, o in [(Path(args.prompts), out), (Path(args.eval_prompts), eval_out)]:
        prompts = load_prompts(prompts_path)
        cache_prompts(prompts, o, enc, args.batch)
        if args.verify:
            verify(prompts, o, enc, args.verify, args.batch)
        write_index(prompts_path, o)
        n = enc.null("")
        torch.save({"emb": n.embeddings[:, : n.mask[0].sum()].to(torch.bfloat16).cpu(),
                    "mask": n.mask[:, : n.mask[0].sum()].cpu()}, o / "null.pt")
        print(f"{o}: null.pt written", flush=True)

    del enc
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()