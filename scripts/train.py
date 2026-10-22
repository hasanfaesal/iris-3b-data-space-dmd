"""Train the data-space generator.

    python scripts/train.py --tiny --out runs/smoke          # CPU smoke test, random tiny model
    python scripts/train.py --out runs/main [--resume ckpt]  # H100, real Iris-3B (needs the embedding cache)
"""

import argparse
import itertools
import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "tests")]

from src.engine.trainer import StdSentinel, TrainConfig, Trainer  # noqa: E402
from src.models.lora import inject_lora  # noqa: E402


def build_tiny():
    from tiny import dataspace_iris, tiny_cond

    model = dataspace_iris()

    def batches(start):  # random stand-in for the cached-embedding loader
        for t in itertools.count(start):
            torch.manual_seed(1000 + t)
            y, mask = tiny_cond(4)
            yield {"y": y, "y_mask": mask, "hw": (16, 16)}

    return model, tiny_cond(1), batches, "cpu", None


def build_real(args):
    """Real Iris-3B on CUDA, per the meta-build order in src/models/dit.py."""
    if not args.cache:
        raise SystemExit("--cache <dir> is required for real runs (cache_embeddings.py output)")
    from src.data.embed_cache import EmbedCache, load_null
    from src.models.loading import load_dataspace_iris

    cache = EmbedCache(args.cache)
    null, null_mask = load_null(args.cache)
    model, _ = load_dataspace_iris("/home/dev/models/iris-3b")
    model.train()  # Iris checkpoints activations only in train mode
    model.set_activation_checkpointing(args.ac, args.ac_every)

    from src.data.prompt_dataset import BucketBatchSampler

    def batches(start):  # infinite stream: epoch-cycling shuffles (resume: skip `start` batches)
        skipped, epoch = start, 0
        while True:
            for k, b in enumerate(BucketBatchSampler(cache.n, seed=epoch)):
                if k >= skipped:
                    y, mask = cache.get(b.indices)
                    yield {"y": y, "y_mask": mask, "hw": b.bucket_hw}
            skipped, epoch = 0, epoch + 1

    return model, (null, null_mask), batches, "cuda", torch.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--checkpoint", default="/home/dev/models/iris-3b")
    ap.add_argument("--cache", default=None, help="embedding cache dir from cache_embeddings.py")
    ap.add_argument("--steps", type=int, default=2400, help="generator updates")
    ap.add_argument("--preview-every", default="250", help="preview frequency in gen updates, or comma-separated steps (e.g. 10,20,40,60)")
    ap.add_argument("--ckpt-every", type=int, default=250, help="checkpoint frequency in gen updates")
    ap.add_argument("--warmup", type=int, default=200, help="fake warmup updates")
    ap.add_argument("--fake-per-gen", type=int, default=4, help="critic updates per generator update")
    ap.add_argument("--overfit", type=int, default=0, help="overfit on first N prompts in cache")
    ap.add_argument(
        "--ac",
        default="full",
        choices=["none", "full", "selective_op", "selective_layer", "pixel_only"],
        help="activation checkpointing (default full; pixel_only OOMs at micro-batch 4 on 80 GB)",
    )
    ap.add_argument("--ac-every", type=int, default=2)
    ap.add_argument("--min-std", type=float, default=0.2, help="sentinel floor; 0.7 * teacher pixel std")
    ap.add_argument("--max-std", type=float, default=0.4, help="sentinel ceiling; abort on static explosion")
    args = ap.parse_args()

    preview_every = (
        {int(x) for x in args.preview_every.split(",")}
        if "," in str(args.preview_every)
        else int(args.preview_every)
    )

    if args.tiny:
        model, (null, null_mask), batches, device, autocast = build_tiny()
        cfg = TrainConfig(warmup=2, fake_per_gen=2, micro_batch=2, num_steps=2, preview_every=2, ckpt_every=2)
        args.steps = min(args.steps, 4)
    else:
        model, (null, null_mask), batches, device, autocast = build_real(args)
        if args.overfit > 0:
            from src.data.embed_cache import EmbedCache
            cache = EmbedCache(args.cache)
            fixed_y, fixed_mask = cache.get(list(range(args.overfit)))
            def overfit_batches(_start):
                while True:
                    yield {"y": fixed_y, "y_mask": fixed_mask, "hw": (1024, 1024)}
            batches = overfit_batches
        cfg = TrainConfig(
            warmup=args.warmup,
            fake_per_gen=args.fake_per_gen,
            preview_every=preview_every,
            ckpt_every=args.ckpt_every,
        )
    inject_lora(model, "gen", 128 if not args.tiny else 8)
    inject_lora(model, "fake", 128 if not args.tiny else 4)

    trainer = Trainer(model, null, null_mask, cfg, device=device, autocast_dtype=autocast, out_dir=args.out)
    if args.resume:
        trainer.load(args.resume)
    stream = batches(trainer.turns)  # resume: skip the batches already consumed
    fixed = next(batches(10**6))  # fixed preview batch
    ok = trainer.fit(stream, args.steps, fixed_batch=fixed, sentinel=StdSentinel(args.min_std, max_std=args.max_std, patience=1))
    trainer.save(Path(args.out) / "checkpoints" / "last.pt")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
