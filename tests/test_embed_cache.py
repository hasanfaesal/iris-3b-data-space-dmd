"""Round-trip test for the cached-embedding loader (CPU, tiny_cond shapes).

Everything is padded to ``length`` (300 in production). Tests use a small length
so the fixture stays tiny; the default is checked separately.
"""

import json

import pytest
import torch

from src.data.embed_cache import TEXT_LEN, EmbedCache, load_null

LENGTH = 10  # stand-in for 300; rows are 1..9 tokens so every row needs padding


@pytest.fixture
def cache(tmp_path):
    # 6 rows as per-prompt files, [T, 32] stand-ins for the real [T, 12, 2560]
    rows = [7, 3, 9, 1, 5, 2]  # per-row token lengths (unpadded)
    for i, t in enumerate(rows):
        torch.save(torch.randn(t, 32), tmp_path / f"row_{i:06d}.pt")
    (tmp_path / "index.json").write_text(json.dumps({"n": len(rows)}))
    torch.save({"emb": torch.randn(1, 4, 32), "mask": torch.ones(1, 4, dtype=torch.int64)},
               tmp_path / "null.pt")
    return tmp_path


def _rows(cache):
    return [torch.load(cache / f"row_{i:06d}.pt", weights_only=True) for i in range(6)]


def test_default_length_is_iris_text_len():
    assert TEXT_LEN == 300 and EmbedCache.__init__.__defaults__[-1] == TEXT_LEN


def test_every_batch_is_padded_to_length(cache):
    c = EmbedCache(cache, length=LENGTH)
    y, mask = c.get([3, 0])  # lengths 1 and 7: both padded to LENGTH, not batch max
    assert y.shape == (2, LENGTH, 32) and mask.shape == (2, LENGTH)
    assert mask.tolist() == [[1] + [0] * 9, [1] * 7 + [0] * 3]


def test_rows_match_files_and_padding_is_zero(cache):
    c = EmbedCache(cache, length=LENGTH)
    rows = _rows(cache)
    for i in (0, 3, 5, 0, 2):  # repeat + out-of-order indices
        y, mask = c.get([i])
        n = rows[i].shape[0]
        assert torch.equal(y[0, :n], rows[i])
        assert torch.equal(y[0, n:], torch.zeros(LENGTH - n, 32))  # padding zeroed, like encode()
        assert mask[0, :n].sum() == n and mask[0, n:].sum() == 0


def test_out_of_range_index(cache):
    with pytest.raises(IndexError):
        EmbedCache(cache, length=LENGTH).get([6])


def test_row_longer_than_length_is_rejected(cache):
    with pytest.raises(ValueError):
        EmbedCache(cache, length=6).get([0])  # row 0 has 7 tokens


def test_load_null_is_padded_to_length(cache):
    null, null_mask = load_null(cache, length=LENGTH)
    assert null.shape == (1, LENGTH, 32) and null_mask.shape == (1, LENGTH)  # [1, ...] for Trainer.expand
    assert null_mask.tolist() == [[1, 1, 1, 1] + [0] * 6]
    assert torch.equal(null[:, 4:], torch.zeros(1, LENGTH - 4, 32))
