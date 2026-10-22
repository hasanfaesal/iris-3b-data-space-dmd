"""Build the training and eval prompt files (Stage 5).

Approved dataset (~25K train prompts):
  - k-mktr/improved-flux-prompts           10K  long descriptive captions
  - data-is-better-together/open-image-preferences-v1  all (~6K)  category-labelled prompts
  - leffff/...Text-Rendering... train       3K  quoted-text rendering
  - NinaKarine/t2i-compbench train         2K  counting / spatial
  - sayakpaul/pick-a-pic-v2-unique-prompts 4K  real user phrasing
Eval (never trained on): 150 PartiPrompts (stratified by category) + 50 leffff test.

Output: data/prompts_train.jsonl, data/prompts_eval.jsonl.

Usage: .venv/bin/python scripts/prepare_prompts.py [--out data]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAX_TOKENS = 256
MIN_WORDS = 5
SEED = 0

# (hf id, config, split, cap, source-name, via). Caps are post-filter targets.
# via="parquet" reads only the text columns over HTTP (image-heavy repo: the
# full dataset would be multi-GB download; this machine has 8 GB RAM).
TRAIN_SOURCES = [
    ("k-mktr/improved-flux-prompts", None, "train", 10_000, "flux", "datasets"),
    ("data-is-better-together/open-image-preferences-v1", "default", "cleaned", None, "image_prefs", "parquet"),
    ("leffff/Diffusion-Reward-Modeling-for-Text-Rendering-Dataset", None, "train", 3_000, "text_render", "datasets"),
    ("NinaKarine/t2i-compbench", None, "train", 2_000, "compbench", "datasets"),
    ("sayakpaul/pick-a-pic-v2-unique-prompts", None, "train", 4_000, "pickapic", "datasets"),
]

_RESIDUE = re.compile(
    r"<lora:|in the style of|artstation|trending on|--ar\b|--v\b|https?://|\{\s*\"|\\\"",
    re.IGNORECASE,
)
_NSFW = re.compile(
    r"\b(nude|naked|nsfw|porn|explicit|erotic|hentai|guro)\b", re.IGNORECASE
)


def normalize_prompt(p: str) -> str:
    return " ".join(p.split())


def is_english(p: str) -> bool:
    return sum(c.isascii() for c in p) / max(len(p), 1) >= 0.9


def is_clean(p: str) -> bool:
    return not _RESIDUE.search(p) and not _NSFW.search(p)


def pick(row: dict, *names: str) -> str | None:
    for n in names:
        v = row.get(n)
        if isinstance(v, str) and v.strip():
            return v
    return None


def iter_prompts_parquet(hf_id: str, config: str, split: str):
    """Text-only rows from the datasets-server parquet conversion, reading only
    the prompt columns — image payloads never download (columnar ranged reads)."""
    import urllib.request

    import fsspec
    import pyarrow.parquet as pq

    url = f"https://huggingface.co/api/datasets/{hf_id}/parquet/{config}/{split}"
    files = json.load(urllib.request.urlopen(url))
    fs = fsspec.filesystem("http")
    for f in files:
        pf = pq.ParquetFile(fs.open(f))
        names = set(pf.schema_arrow.names)
        cols = [c for c in ("prompt", "Prompt", "quality_prompt", "simplified_prompt", "category") if c in names]
        for batch in pf.iter_batches(columns=cols, batch_size=1000):
            for row in batch.to_pylist():
                p = pick(row, "prompt", "Prompt", "quality_prompt", "simplified_prompt")
                if p:
                    yield p, row.get("category")


def iter_prompts(hf_id: str, config: str | None, split: str):
    """Yield (prompt, category) rows from a HF dataset, tolerant of schema variants."""
    from datasets import load_dataset

    try:
        ds = load_dataset(hf_id, config, split=split)
    except (ValueError, KeyError):
        # config-per-category datasets: take the split of every config
        from datasets import get_dataset_config_names

        for cfg in get_dataset_config_names(hf_id):
            try:
                part = load_dataset(hf_id, cfg, split=split)
            except (ValueError, KeyError):
                continue
            for row in part:
                p = pick(row, "prompt", "Prompt", "simplified_prompt", "quality_prompt")
                if p:
                    yield p, row.get("category") or cfg
        return
    for row in ds:
        p = pick(row, "prompt", "Prompt", "quality_prompt", "simplified_prompt")
        if p:
            yield p, row.get("category")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-VL-4B-Instruct")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print("loading tokenizer…")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def fits_budget(p: str) -> bool:
        return MIN_WORDS <= len(p.split()) and len(tok(p).input_ids) <= MAX_TOKENS

    # ---- eval set first: Parti (150, stratified) + leffff test (50) ----
    from datasets import load_dataset

    parti = list(load_dataset("nateraw/parti-prompts", split="train"))
    by_cat: dict[str, list[dict]] = {}
    for row in parti:
        p = pick(row, "Prompt", "prompt")
        if not p:
            raise SystemExit(f"parti-prompts: unexpected columns {list(row)}")
        by_cat.setdefault(row.get("Category", "none"), []).append(row)
    eval_rows = []
    take = 150
    for cat, rows in sorted(by_cat.items()):
        n = max(1, round(take * len(rows) / len(parti)))
        eval_rows += [
            {"prompt": normalize_prompt(r["Prompt"]), "source": "parti", "category": cat}
            for r in rows[:n]
        ]
    eval_rows = eval_rows[:take]

    leffff_test = list(load_dataset(TRAIN_SOURCES[2][0], split="test"))
    for row in leffff_test:
        if len(eval_rows) >= 200:
            break
        p = pick(row, "prompt", "Prompt")
        if p:
            eval_rows.append({"prompt": normalize_prompt(p), "source": "text_render", "category": "text"})

    eval_norm = {r["prompt"] for r in eval_rows}
    with (out / "prompts_eval.jsonl").open("w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    print(f"eval: {len(eval_rows)} prompts "
          f"({Counter(r['source'] for r in eval_rows)})")

    # ---- training pool ----
    train_rows: list[dict] = []
    seen: set[str] = set()
    report: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    for hf_id, config, split, cap, name, via in TRAIN_SOURCES:
        pool = []
        rows = iter_prompts_parquet(hf_id, config, split) if via == "parquet" else iter_prompts(hf_id, config, split)
        for p, _cat in rows:
            p = normalize_prompt(p)
            if p in seen or p in eval_norm:
                dropped[f"{name}:dup"] += 1
                continue
            if not is_english(p) or not is_clean(p) or not fits_budget(p):
                dropped[f"{name}:filter"] += 1
                continue
            pool.append(p)
        rng.shuffle(pool)
        keep = pool if cap is None else pool[:cap]
        seen.update(keep)
        train_rows += [{"prompt": p, "source": name} for p in keep]
        report[name] = len(keep)
        print(f"{name}: kept {len(keep)} (of {len(pool)} passing filters)")

    rng.shuffle(train_rows)
    with (out / "prompts_train.jsonl").open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")

    lens = sorted(len(r["prompt"].split()) for r in train_rows)
    n = len(lens)
    print(f"train: {n} prompts, sources={dict(report)}")
    print(f"dropped: {dict(dropped)}")
    print(f"words: min={lens[0]} median={lens[n//2]} max={lens[-1]}")
    quoted = sum('"' in r["prompt"] for r in train_rows)
    print(f"quoted-text prompts: {quoted} ({quoted / n:.1%})")


if __name__ == "__main__":
    main()
