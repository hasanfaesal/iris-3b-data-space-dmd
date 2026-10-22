"""Prompt dataset and aspect-bucket batch sampler.

Prompts are stored as JSONL rows {"prompt": str, "source": str} (train) plus
{"prompt": str, "source": str, "category": str} (eval). Text is embedded
elsewhere; this module only serves prompts and groups them into resolution
buckets. One aspect bucket is assigned per global batch (IMPLEMENT.md 3.1).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, NamedTuple

# ~1 MP buckets, patch size 16 aligned (IMPLEMENT.md 3.1).
PROMPT_BUCKETS: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (768, 1344),
    (896, 1152),
    (1344, 768),
    (1152, 896),
)


class Batch(NamedTuple):
    bucket_hw: tuple[int, int]
    indices: list[int]


class PromptDataset:
    """Reads a prompts JSONL file; row order is the index."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows: list[dict] = []
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"no prompts in {self.path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


class BucketBatchSampler:
    """Yields (bucket, indices) per step — one aspect bucket per global batch.

    Prompt order is shuffled per epoch with a seedable RNG; the bucket is drawn
    uniformly per batch. Every prompt appears exactly once per epoch.
    """

    def __init__(
        self,
        dataset_len: int,
        global_batch: int = 16,
        buckets: tuple[tuple[int, int], ...] = PROMPT_BUCKETS,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if global_batch < 1:
            raise ValueError("global_batch must be >= 1")
        self.dataset_len = dataset_len
        self.global_batch = global_batch
        self.buckets = tuple(buckets)
        self.seed = seed
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = self.dataset_len // self.global_batch
        if not self.drop_last and self.dataset_len % self.global_batch:
            n += 1
        return n

    def __iter__(self) -> Iterator[Batch]:
        rng = random.Random(self.seed)
        order = list(range(self.dataset_len))
        rng.shuffle(order)
        for start in range(0, self.dataset_len, self.global_batch):
            batch = order[start : start + self.global_batch]
            if self.drop_last and len(batch) < self.global_batch:
                break
            yield Batch(rng.choice(self.buckets), batch)
