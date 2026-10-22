"""Round-trip smoke test for the cached-embedding loader (CPU, tiny_cond shapes)."""

import json

import pytest
import torch

from src.data.embed_cache import EmbedCache, load_null


@pytest.fixture
def cache(tmp_path):
    # 3 shards of 2 rows, tiny_cond-shaped: [T, 12, 2560] is real-size; use [T, 32] like tiny
    rows = [7, 3, 9, 1, 5, 2]  # per-row token lengths (unpadded)
    shard_size = 2
    for s in range(3):
        part = rows[s * shard_size : (s + 1) * shard_size]
        torch.save({"rows": [torch.randn(t, 32) for t in part]}, tmp_path / f"shard_{s:04d}.pt")
    (tmp_path / "index.json").write_text(json.dumps({"n": len(rows), "shard_size": shard_size}))
    torch.save({"emb": torch.randn(1, 4, 32), "mask": torch.ones(1, 4, dtype=torch.int64)},
               tmp_path / "null.pt")
    return tmp_path


def test_roundtrip_matches_shard_rows(cache):
    c = EmbedCache(cache, lru=2)  # small LRU to force reloads across the boundary
    rows = [r for s in range(3) for r in torch.load(cache / f"shard_{s:04d}.pt", weights_only=True)["rows"]]
    for i in (0, 3, 5, 0, 2):  # repeat + shard-spanning indices
        got, _ = c.get([i])
        assert got.shape == (1, rows[i].shape[0], 32)
        assert torch.equal(got[0], rows[i])


def test_pads_to_batch_max_with_zero_and_mask(cache):
    c = EmbedCache(cache)
    y, mask = c.get([3, 0])  # lengths 1 and 7 -> batch max 7
    assert y.shape == (2, 7, 32) and mask.shape == (2, 7)
    assert mask.tolist() == [[1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]]
    assert torch.equal(y[0, 1:], torch.zeros(6, 32))  # padding zeroed, like encode()
    rows = [r for s in range(3) for r in torch.load(cache / f"shard_{s:04d}.pt", weights_only=True)["rows"]]
    assert torch.equal(y[1], rows[0]) and torch.equal(y[0, :1], rows[3])


def test_out_of_range_index(cache):
    with pytest.raises(IndexError):
        EmbedCache(cache).get([6])


def test_load_null(cache):
    null, null_mask = load_null(cache)
    assert null.shape == (1, 4, 32) and null_mask.shape == (1, 4)  # [1, ...] for Trainer.expand
