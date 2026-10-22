"""Loader for the offline Qwen embedding cache written by scripts/cache_embeddings.py.

Cache layout: one row_XXXXXX.pt file per prompt holding an unpadded [T, C, D] bf16 tensor
(T = true token count); index.json ({n}); null.pt ({emb [1, T, C, D], mask [1, T]}).

Everything served here is padded to ``length`` (Iris text_len = 300) with zeros and mask 0,
exactly like ``encode()``. The trunk's joint attention does not mask pad tokens, so the
teacher only sees the same conditioning as sampling if the length is always 300.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

TEXT_LEN = 300  # config.py text_len; encode() always pads to max_length=300


class EmbedCache:
    """Serves cached embeddings by row index; get(indices) pads to ``length``."""

    def __init__(self, cache_dir: str | Path, length: int = TEXT_LEN):
        self.dir, self.length = Path(cache_dir), length
        self.n = json.loads((self.dir / "index.json").read_text())["n"]

    def get(self, indices) -> tuple[torch.Tensor, torch.Tensor]:
        """(y [B, length, C, D] bf16, y_mask [B, length] int64); padding zeroed like encode()."""
        if not all(0 <= i < self.n for i in indices):
            raise IndexError(f"indices out of range 0..{self.n - 1}: {indices}")
        rows = [torch.load(self.dir / f"row_{i:06d}.pt", weights_only=True) for i in indices]
        y = torch.zeros(len(rows), self.length, *rows[0].shape[1:], dtype=rows[0].dtype)
        mask = torch.zeros(len(rows), self.length, dtype=torch.int64)
        for i, r in enumerate(rows):
            if r.shape[0] > self.length:
                raise ValueError(f"row has {r.shape[0]} tokens, cache length is {self.length}")
            y[i, : r.shape[0]] = r
            mask[i, : r.shape[0]] = 1
        return y, mask


def load_null(cache_dir: str | Path, length: int = TEXT_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    """(null [1, length, C, D], null_mask [1, length]) for the Trainer constructor."""
    d = torch.load(Path(cache_dir) / "null.pt", weights_only=True)
    emb, mask = d["emb"], d["mask"]
    if emb.shape[1] > length:
        raise ValueError(f"null has {emb.shape[1]} tokens, cache length is {length}")
    null = torch.zeros(1, length, *emb.shape[2:], dtype=emb.dtype)
    null_mask = torch.zeros(1, length, dtype=mask.dtype)
    null[:, : emb.shape[1]] = emb
    null_mask[:, : mask.shape[1]] = mask
    return null, null_mask
