"""Train the data-space generator.

    python scripts/train.py --tiny --out runs/smoke          # CPU smoke test, random tiny model
    python scripts/train.py --out runs/main [--resume ckpt]  # H100, real Iris-3B (needs the embedding cache)
"""

import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "tests")]

from src.engine.trainer import StdSentinel, TrainConfig, Trainer  # noqa: E402
from src.models.lora import inject_lora  # noqa: E402

# Concrete held-out Parti rows. prompts_eval.jsonl starts with one-word abstract
# prompts; those are a bad std baseline. Indices stay < 100.
PREVIEW_ROWS = (
    (5, "a shiba inu"),
    (11, "a corgi"),
    (18, "a cat"),
    (23, "a horse"),
    (36, "a chair"),
    (46, "a laptop"),
    (54, "a coffee mug"),
    (66, "A bowl of Pho"),
    (72, "a roast turkey"),
    (84, "a kitchen"),
    (87, "a living room"),
    (88, "a fall landscape"),
    (90, "a city"),
    (94, "a street"),
    (95, "a house"),
    (96, "a city intersection"),
)
COLLAPSE_AFTER = 100  # v4 mean std was still ~0.19 at update 50


def log_disk(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    print(f"disk free {free / 2**30:.1f} GiB at {path}", flush=True)
    # last.pt ~7 GiB (needs a second copy while writing) + ~16 GiB of slim snapshots + previews.
    if free < 40 << 30:
        print(
            "warning: under 40 GiB free. Refreshing last.pt needs ~7 GiB free on top of the "
            "file itself; nine slim snapshots are ~16 GiB. A save that would fill the disk is skipped.",
            flush=True,
        )
    subprocess.run(["df", "-h", str(path)], check=False)


def drop_diagnostic_runs(runs: Path, keep: Path) -> None:
    """Delete leftover ``runs/diagnostic*`` trees. ``keep`` is the directory this launch writes."""
    if not runs.is_dir():
        return
    keep = keep.resolve()
    for p in list(runs.glob("diagnostic*")):
        resolved = p.resolve()
        if resolved == keep or keep.is_relative_to(resolved):
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"removed {p}", flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_preview_batch(eval_cache, prompts_path: Path):
    """Fixed eval rows. Exits if the cache was built from a different prompt file."""
    if not prompts_path.is_file():
        raise SystemExit(f"missing {prompts_path}")
    index_path = Path(eval_cache.dir) / "index.json"
    index = json.loads(index_path.read_text())
    digest = _sha256(prompts_path)
    if index.get("sha256") != digest:
        raise SystemExit(
            f"{eval_cache.dir} was not built from {prompts_path} "
            f"(cache {index.get('sha256')} != file {digest}). Recache eval embeddings."
        )
    rows = [json.loads(line)["prompt"] for line in prompts_path.read_text().splitlines() if line.strip()]
    bad = [f"{i}:{text}" for i, text in PREVIEW_ROWS if i >= eval_cache.n or i >= len(rows) or rows[i] != text]
    if bad:
        raise SystemExit(f"preview rows missing or moved: {bad}")
    y, mask = eval_cache.get([i for i, _ in PREVIEW_ROWS])
    return {"y": y, "y_mask": mask, "hw": (1024, 1024)}


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
        epoch, offset = divmod(start, len(BucketBatchSampler(cache.n)))
        while True:
            for k, b in enumerate(BucketBatchSampler(cache.n, seed=epoch)):
                if k >= offset:
                    y, mask = cache.get(b.indices)
                    yield {"y": y, "y_mask": mask, "hw": b.bucket_hw}
            offset, epoch = 0, epoch + 1

    return model, (null, null_mask), batches, "cuda", torch.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--checkpoint", default="/home/dev/models/iris-3b")
    ap.add_argument("--cache", default=None, help="embedding cache dir from cache_embeddings.py")
    ap.add_argument("--eval-cache", default=None, help="eval embedding cache dir (defaults to <cache>_eval)")
    ap.add_argument("--steps", type=int, default=2400, help="generator updates")
    ap.add_argument("--preview-every", default="50", help="preview frequency in gen updates, or comma-separated steps (e.g. 10,20,40,60)")
    ap.add_argument("--ckpt-every", type=int, default=250, help="slim gen+EMA snapshot every N gen updates (0 disables)")
    ap.add_argument("--rolling-every", type=int, default=100, help="overwrite checkpoints/last.pt every N gen updates (0 disables)")
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
    ap.add_argument("--min-std", type=float, default=None, help="sentinel floor (default: 0.6 * teacher pixel std)")
    ap.add_argument("--std-ratio", type=float, default=0.6, help="sentinel collapse ratio relative to teacher std")
    ap.add_argument("--max-std", type=float, default=0.4, help="sentinel ceiling; abort on static explosion")
    ap.add_argument(
        "--deadline",
        default="2026-10-26T06:53:00+00:00",
        help="UTC deadline ISO string or epoch seconds to save and exit cleanly before credit expiry",
    )
    ap.add_argument(
        "--events-interval",
        type=float,
        default=120.0,
        help="seconds between Givemeanode /v1/events checks (0 to disable)",
    )
    args = ap.parse_args()
    out = Path(args.out)
    if not args.tiny:
        drop_diagnostic_runs(Path(__file__).resolve().parents[1] / "runs", out)
    log_disk(out)

    preview_every = (
        {int(x) for x in args.preview_every.split(",")}
        if "," in str(args.preview_every)
        else int(args.preview_every)
    )

    if args.tiny:
        model, (null, null_mask), batches, device, autocast = build_tiny()
        cfg = TrainConfig(warmup=2, fake_per_gen=2, micro_batch=2, num_steps=2, preview_every=2, ckpt_every=2)
        args.steps = min(args.steps, 4)
        fixed = next(batches(10**6))
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
            fixed = {"y": fixed_y, "y_mask": fixed_mask, "hw": (1024, 1024)}
        else:
            from src.data.embed_cache import EmbedCache
            eval_dir = Path(args.eval_cache) if args.eval_cache else (Path(args.cache).parent / (Path(args.cache).name + "_eval"))
            if not (eval_dir / "index.json").is_file():
                raise SystemExit(f"eval embedding cache not found at {eval_dir}. Refusing to preview a training batch.")
            fixed = load_preview_batch(
                EmbedCache(str(eval_dir)), Path(__file__).resolve().parents[1] / "data" / "prompts_eval.jsonl"
            )
        cfg = TrainConfig(
            warmup=args.warmup,
            fake_per_gen=args.fake_per_gen,
            preview_every=preview_every,
            ckpt_every=args.ckpt_every,
            rolling_every=args.rolling_every,
        )
    inject_lora(model, "gen", 128 if not args.tiny else 8)
    inject_lora(model, "fake", 128 if not args.tiny else 4)

    trainer = Trainer(model, null, null_mask, cfg, device=device, autocast_dtype=autocast, out_dir=args.out)
    if args.resume:
        trainer.load(args.resume)

    teacher_std = None
    if not args.tiny:
        print("Rendering teacher preview baseline...", flush=True)
        teacher_std = trainer.render_teacher(fixed)
        print(f"Teacher preview pixel std: {teacher_std:.4f} (cfg={cfg.cfg_scale})", flush=True)

    sentinel = StdSentinel(
        min_std=args.min_std,
        max_std=args.max_std,
        patience=2,
        teacher_std=teacher_std,
        ratio=args.std_ratio,
        collapse_after=0 if args.tiny else COLLAPSE_AFTER,
    )
    print(
        f"StdSentinel active: bounds=[{sentinel.min_std:.4f}, {sentinel.max_std:.4f}], "
        f"patience={sentinel.patience}, collapse_after={sentinel.collapse_after}"
        + (f" (teacher_std={teacher_std:.4f})" if teacher_std is not None else ""),
        flush=True,
    )
    print(
        "rollback: first static -> gen lr 5e-6, first collapse or non-finite loss -> gen lr 1e-5, "
        "fake lr stays 5e-5, restore checkpoints/last_good.pt; a second trip stops",
        flush=True,
    )

    from src.utils.platform import PlatformSentinel, parse_deadline

    deadline_ts = parse_deadline(args.deadline)
    platform_sentinel = PlatformSentinel(deadline=deadline_ts, check_interval=args.events_interval)
    if deadline_ts:
        print(f"PlatformSentinel active: deadline={args.deadline} (ts={deadline_ts}), events_interval={args.events_interval}s", flush=True)

    stream = batches(trainer.turns)  # resume: skip the batches already consumed
    ok = trainer.fit(
        stream,
        args.steps,
        fixed_batch=fixed,
        sentinel=sentinel,
        platform_sentinel=platform_sentinel,
    )
    if ok:
        trainer._save_full()
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
