"""Loader for the offline Qwen embedding cache written by scripts/cache_embeddings.py.

Cache layout (see cache_embeddings.py): shard_XXXX.pt files holding {"rows": [...]}
in row order, SHARD rows per shard, each row an unpadded [T, C, D] bf16 tensor;
index.json ({n, shard_size}); null.pt ({emb [1, T, C, D], mask [1, T]}).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


class EmbedCache:
    """Serves cached embeddings by row index; get(indices) pads to the batch max."""

    def __init__(self, cache_dir: str | Path, lru: int = 4):
        self.dir = Path(cache_dir)
        index = json.loads((self.dir / "index.json").read_text())
        self.n, self.shard_size = index["n"], index["shard_size"]
        self._shards: dict[int, list] = {}
        self._lru: list[int] = []
        self._lru_max = lru

    def _shard(self, s: int) -> list:
        if s in self._shards:
            self._lru.remove(s)
            self._lru.append(s)
            return self._shards[s]
        rows = torch.load(self.dir / f"shard_{s:04d}.pt", weights_only=True)["rows"]
        self._shards[s], _ = rows, self._lru.append(s)
        if len(self._lru) > self._lru_max:
            self._shards.pop(self._lru.pop(0))
        return rows

    def get(self, indices) -> tuple[torch.Tensor, torch.Tensor]:
        """(y [B, Lmax, C, D] bf16, y_mask [B, Lmax] int64); padding zeroed like encode()."""
        if not all(0 <= i < self.n for i in indices):
            raise IndexError(f"indices out of range 0..{self.n - 1}: {indices}")
        rows = [self._shard(i // self.shard_size)[i % self.shard_size] for i in indices]
        lmax = max(r.shape[0] for r in rows)
        y = torch.zeros(len(rows), lmax, *rows[0].shape[1:], dtype=rows[0].dtype)
        mask = torch.zeros(len(rows), lmax, dtype=torch.int64)
        for i, r in enumerate(rows):
            y[i, : r.shape[0]] = r
            mask[i, : r.shape[0]] = 1
        return y, mask


def load_null(cache_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """(null [1, T, C, D], null_mask [1, T]) for the Trainer constructor."""
    d = torch.load(Path(cache_dir) / "null.pt", weights_only=True)
    return d["emb"], d["mask"]
