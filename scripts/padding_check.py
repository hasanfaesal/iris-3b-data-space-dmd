"""Padding-convention check for the Qwen embedding cache (H100, ~5 min; `--tiny` = CPU dry run).

The teacher was trained with text ALWAYS padded to text_len (300): pad positions are not
masked in the trunk's joint attention (only y_embedder masks them), so they carry
y_pos_embedding-derived values and the trunk softmaxes over 300 text keys. The cache
instead stores rows cut to true length, EmbedCache.get pads only to the batch max, and
null.pt is cut - so training currently shows the trunk fewer text keys than the teacher
ever saw, and the count varies per batch.

For a batch of prompts this compares the teacher velocity (adapters off, eval, no grad):

  A "pad300": cond AND null zero-padded to text_len, mask 0 on pads  (encoder-native)
  B "cut":    cond padded to batch max, null at its true length      (current cache/training)

at model time t in {1000, 600, 200} (sigma' = t/1000). The CFG window (sigma' <= 0.6)
makes v_cfg operative at t=600/200 and plain v_cond at t=1000. Inputs (z, x_tau) are
identical for A and B, so any difference is the convention. The t=1000 rel err also
equals the relative divergence of the first rollout step's x0 estimate (x0_hat = z - v).

Gate (decided on the fp32 run, which is free of the ~1.4% bf16 autocast noise floor from
Stage 7 parity): worst batch-mean relative L2 of the field training actually uses, over
t and prompt groups. Above --tol (default 1%) -> apply the padding fix.

  python -u scripts/padding_check.py --cache /home/dev/cache --checkpoint /home/dev/models/iris-3b --out runs/padding_check
  python -u scripts/padding_check.py --tiny --out /tmp/padding_check     # CPU logic check, no cache needed

Everything printed also lands in <out>/padding_check.log. Exit 0 = pass, 1 = gate trips.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

T_VALUES = (1000.0, 600.0, 200.0)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/home/dev/cache", help="embedding cache dir (cache_embeddings.py output)")
    ap.add_argument("--checkpoint", default="/home/dev/models/iris-3b")
    ap.add_argument("--prompts", default=str(ROOT / "data" / "prompts_train.jsonl"),
                    help="jsonl the cache rows were built from (row i = line i); for the log only")
    ap.add_argument("--n", type=int, default=16, help="prompts per group; 'first' group = the overfit set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.01, help="gate on worst batch-mean relative L2 of the used field")
    ap.add_argument("--cfg-scale", type=float, default=3.0)
    ap.add_argument("--cfg-window", type=float, default=0.6, help="CFG where shifted sigma' <= window (losses.py)")
    ap.add_argument("--tiny", action="store_true", help="CPU logic check on the tiny fixture (no cache/checkpoint)")
    return ap.parse_args()


def pad_to(y: torch.Tensor, mask: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad text states [B, T, ...] + mask to `length` (encoder-native convention)."""
    if y.shape[1] > length:
        raise ValueError(f"text length {y.shape[1]} exceeds pad target {length}")
    y_pad = torch.zeros(y.shape[0], length, *y.shape[2:], dtype=y.dtype, device=y.device)
    y_pad[:, : y.shape[1]] = y
    m = torch.zeros(mask.shape[0], length, dtype=mask.dtype, device=mask.device)
    m[:, : mask.shape[1]] = mask
    return y_pad, m


def rel_err_per_row(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.float(), b.float()
    return (a - b).flatten(1).norm(dim=1) / a.flatten(1).norm(dim=1)


def prompt_texts(path: str, indices, width: int = 70):
    """Truncated prompt text per cache row (row i = jsonl line i), or None if unreadable."""
    try:
        lines = Path(path).read_text().splitlines()
        out = []
        for i in indices:
            try:
                out.append(json.loads(lines[i])["prompt"][:width])
            except (IndexError, KeyError, ValueError):
                return None
        return out
    except OSError:
        return None


class FakeCache:
    """Tiny-mode stand-in for EmbedCache: rows cut to varied true lengths (< text_len)."""

    def __init__(self, n: int = 32, seed: int = 0):
        from tiny import TINY_TEXT_DIM, TINY_TEXT_LEN

        g = torch.Generator().manual_seed(seed)
        base = torch.randn(n, TINY_TEXT_LEN, TINY_TEXT_DIM, generator=g)
        self.n = n
        self.rows = [base[i, : int(torch.randint(3, TINY_TEXT_LEN - 2, (1,), generator=g))] for i in range(n)]

    def get(self, indices) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [self.rows[i] for i in indices]
        lmax = max(r.shape[0] for r in rows)
        y = torch.zeros(len(rows), lmax, rows[0].shape[1])
        mask = torch.zeros(len(rows), lmax, dtype=torch.int64)
        for k, r in enumerate(rows):
            y[k, : r.shape[0]], mask[k, : r.shape[0]] = r, 1
        return y, mask

    def null(self) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.rows[0][:5]
        return r[None], torch.ones(1, 5, dtype=torch.int64)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "padding_check.log"
    t_start = time.perf_counter()

    def log(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    device = torch.device("cpu" if args.tiny else "cuda")
    g = torch.Generator().manual_seed(args.seed)
    log(f"padding_check {time.strftime('%Y-%m-%d %H:%M:%S')} | torch {torch.__version__} | "
        f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}")
    log(f"args: {vars(args)}")

    if args.tiny:
        from tiny import dataspace_iris

        model = dataspace_iris(seed=args.seed)
        cache = FakeCache(seed=args.seed + 1)
        null_cut, null_mask_cut = cache.null()
        hw = (16, 16)
    else:
        from src.data.embed_cache import EmbedCache, load_null
        from src.models.loading import load_dataspace_iris

        model, mcfg = load_dataspace_iris(args.checkpoint)
        cache = EmbedCache(args.cache)
        null_cut, null_mask_cut = load_null(args.cache)
        hw = (1024, 1024)
        log(f"model: text_len={mcfg.text_len} text_adapter={mcfg.text_adapter} "
            f"hidden={mcfg.hidden_size} depth={mcfg.depth} patch={mcfg.patch_size}")
        log(f"cache: n={cache.n} rows in {args.cache}")
    text_len = model.cfg.text_len
    log(f"null: true length T_null={null_cut.shape[1]} (cut) vs {text_len} (pad300); "
        f"pad positions in the trunk are NOT masked, only in y_embedder")
    if args.tiny:
        log("NOTE: tiny fixture = random weights, logic check ONLY - the numbers are NOT meaningful")

    n = args.n
    groups = [
        (f"first{n} (overfit set)", list(range(n))),
        (f"rand{n}", torch.randperm(cache.n, generator=g)[: n].tolist()),
    ]
    for gname, idx in groups:
        texts = None if args.tiny else prompt_texts(args.prompts, idx)
        log(f"group '{gname}': rows {idx if len(idx) <= 8 else idx[:8] + ['...']}")
        if texts:
            for i, (row, text) in enumerate(zip(idx, texts)):
                log(f"  row {row:>6}: {text}")

    # modes: bf16 first (= exactly what training runs), then fp32 (clean, decides the gate)
    modes = [("bf16", True), ("fp32", False)] if device.type == "cuda" else [("fp32", False)]
    per_mode: dict[str, float] = {}

    for dtype_name, use_ac in modes:
        run_dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
        if dtype_name == "fp32":
            model.to(torch.float32)  # upcast in place; bf16 mode used the model exactly as loaded
        autocast_ctx = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_ac)  # noqa: E731
        per_mode[dtype_name] = (0.0, "")
        try:
            for gi, (gname, idx) in enumerate(groups):
                y_cut, mask_cut = cache.get(idx)
                y_cut = y_cut.to(device=device, dtype=run_dtype)
                mask_cut = mask_cut.to(device)
                y_pad, mask_pad = pad_to(y_cut, mask_cut, text_len)
                null_y = null_cut.to(device=device, dtype=run_dtype)
                null_m = null_mask_cut.to(device)
                null_pad, null_mask_pad = pad_to(null_y, null_m, text_len)
                batch = y_cut.shape[0]

                def forward(x, t, y, m):
                    with torch.no_grad(), autocast_ctx():
                        return model(x, t, y, y_mask=m).x.float()

                log(f"== dtype={dtype_name} group='{gname}' ==")
                lens = mask_cut.sum(1).tolist()
                log(f"  tokens per prompt: {lens} | batch max {max(lens)} | null {null_y.shape[1]} "
                    f"| pad300 target {text_len}")

                z = torch.randn(batch, 3, *hw, generator=g).to(device)
                with autocast_ctx():
                    v0 = model(z, torch.full((batch,), 1000.0, device=device), y_cut, y_mask=mask_cut).x
                x0_hat = z - v0.float()  # convention B: matches current training's rollout step 1
                log("  inputs: z fixed; x0_hat = z - v_cond_B(z, t=1000) (convention B); "
                    "x_tau(t<1000) = (1-s)x0_hat + s*eps; identical for A and B")

                worst_t = {}
                for t_val in T_VALUES:
                    s = t_val / 1000.0
                    if t_val >= 1000.0:
                        x_tau = z
                    else:
                        eps = torch.randn(z.shape, generator=g).to(device)
                        x_tau = (1 - s) * x0_hat + s * eps
                    t_t = torch.full((batch,), t_val, device=device)
                    cfg_on = s <= args.cfg_window
                    cond = {c: forward(x_tau, t_t, y, m) for c, (y, m) in
                            (("pad300", (y_pad, mask_pad)), ("cut", (y_cut, mask_cut)))}
                    rel_cond = rel_err_per_row(cond["pad300"], cond["cut"])
                    if cfg_on:
                        unc = {c: forward(x_tau, t_t, y.expand(batch, *y.shape[1:]), m.expand(batch, -1))
                               for c, (y, m) in (("pad300", (null_pad, null_mask_pad)), ("cut", (null_y, null_m)))}
                        cfg = {c: unc[c] + args.cfg_scale * (cond[c] - unc[c]) for c in cond}
                        rel_cfg = rel_err_per_row(cfg["pad300"], cfg["cut"])
                        used = rel_cfg
                        field = f"v_cfg(scale {args.cfg_scale}, window ON)"
                    else:
                        rel_cfg, used, field = None, rel_cond, "v_cond (window OFF)"
                    worst_t[t_val] = used.mean().item()

                    log(f"  t={t_val:.0f} sigma'={s:.3f} field={field}")
                    for i in range(batch):
                        line = f"    row {i:>3} tok {lens[i]:>3}  rel(cond) {rel_cond[i]:.4f}"
                        line += f"  rel(cfg) {rel_cfg[i]:.4f}" if rel_cfg is not None else "  rel(cfg)    n/a"
                        log(line)
                    line = f"    mean      rel(cond) {rel_cond.mean():.4f}"
                    line += f"  rel(cfg) {rel_cfg.mean():.4f}" if rel_cfg is not None else ""
                    line += f"  | used-field mean {used.mean():.4f}  worst row {used.max():.4f}"
                    log(line)
                group_worst = max(worst_t.values())
                worst_at = max(worst_t, key=worst_t.get)
                if group_worst > per_mode[dtype_name][0]:
                    per_mode[dtype_name] = (group_worst, f"group '{gname}', t={worst_at:.0f}")
                log(f"  dtype={dtype_name} group='{gname}': worst used-field batch-mean rel err over t "
                    f"= {group_worst:.4f}")
        except torch.OutOfMemoryError:
            log(f"  !! OOM during dtype={dtype_name}; skipping the rest of this mode")
            per_mode.pop(dtype_name, None)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        if device.type == "cuda":
            torch.cuda.empty_cache()

    decision = "fp32" if "fp32" in per_mode else "bf16"
    worst, where = per_mode[decision]
    ok = worst <= args.tol
    log("")
    log("== summary ==")
    for d, (w, at) in per_mode.items():
        log(f"  {d}: worst used-field batch-mean rel err (over t, groups) = {w:.4f}  [{at}]")
    log(f"bf16 rows include the ~1.4% bf16 autocast noise floor (Stage 7 parity: 0.0139); "
        f"the gate is decided on {decision}.")
    log(f"GATE: worst = {worst:.4f} vs tol {args.tol} -> {'PASS' if ok else 'FAIL'}")
    if ok:
        log("  padding convention is within tolerance; no cache change needed")
    else:
        log(f"  APPLY the padding fix: pad cond batches AND the CFG null to text_len={text_len} "
            "with zeros + mask 0 - pad null.pt at cache-build time and pad batches in EmbedCache.get")
    log(f"elapsed {time.perf_counter() - t_start:.0f}s | log: {log_path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()