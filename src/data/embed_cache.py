"""Loader for the offline Qwen embedding cache written by scripts/cache_embeddings.py.

Cache layout: one row_XXXXXX.pt file per prompt holding an unpadded [T, C, D] bf16 tensor
(T = true token count); index.json ({n}); null.pt ({emb [1, T, C, D], mask [1, T]}).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


class EmbedCache:
    """Serves cached embeddings by row index; get(indices) pads to the batch max."""

    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self.n = json.loads((self.dir / "index.json").read_text())["n"]

    def get(self, indices) -> tuple[torch.Tensor, torch.Tensor]:
        """(y [B, Lmax, C, D] bf16, y_mask [B, Lmax] int64); padding zeroed like encode()."""
        if not all(0 <= i < self.n for i in indices):
            raise IndexError(f"indices out of range 0..{self.n - 1}: {indices}")
        rows = [torch.load(self.dir / f"row_{i:06d}.pt", weights_only=True) for i in indices]
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