import json

import pytest

from scripts.prepare_prompts import is_clean, is_english, normalize_prompt
from src.data.prompt_dataset import PROMPT_BUCKETS, BucketBatchSampler, PromptDataset


def write_jsonl(tmp_path, rows, name="prompts_train.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def dataset(tmp_path):
    rows = [{"prompt": f"prompt {i}", "source": "test"} for i in range(50)]
    return PromptDataset(write_jsonl(tmp_path, rows))


def test_buckets_are_one_mp_and_patch_aligned():
    for h, w in PROMPT_BUCKETS:
        assert h % 16 == 0 and w % 16 == 0
        assert 0.9e6 <= h * w <= 1.1e6


def test_dataset_reads_rows(dataset):
    assert len(dataset) == 50
    assert dataset[0]["prompt"] == "prompt 0"


def test_dataset_rejects_empty_file(tmp_path):
    with pytest.raises(ValueError):
        PromptDataset(write_jsonl(tmp_path, []))


def test_one_bucket_per_batch(dataset):
    sampler = BucketBatchSampler(len(dataset), global_batch=16, seed=0)
    seen = []
    for batch in sampler:
        assert isinstance(batch.bucket_hw, tuple)
        assert 1 <= len(batch.indices) <= 16
        assert batch.bucket_hw in PROMPT_BUCKETS
        seen += batch.indices
    assert sorted(seen) == list(range(50))  # every prompt exactly once per epoch


def test_partial_last_batch_size(dataset):
    sampler = BucketBatchSampler(len(dataset), global_batch=16, seed=0)
    sizes = [len(b.indices) for b in sampler]
    assert sizes == [16, 16, 16, 2]


def test_drop_last(dataset):
    sampler = BucketBatchSampler(len(dataset), global_batch=16, seed=0, drop_last=True)
    assert [len(b.indices) for b in sampler] == [16, 16, 16]


def test_epoch_deterministic(dataset):
    def epoch(seed):
        return [(b.bucket_hw, b.indices) for b in BucketBatchSampler(len(dataset), seed=seed)]
    assert epoch(0) == epoch(0)
    assert len(epoch(1)) == len(epoch(2))


def test_batches_resume_across_epochs():
    # Verify stream resumption across multiple epoch boundaries matches full un-resumed run
    n = 50
    sampler_len = len(BucketBatchSampler(n))

    def make_stream(start):
        epoch, offset = divmod(start, sampler_len)
        while True:
            for k, b in enumerate(BucketBatchSampler(n, seed=epoch)):
                if k >= offset:
                    yield (epoch, k, b.indices)
            offset, epoch = 0, epoch + 1

    stream = make_stream(0)
    full = [next(stream) for _ in range(sampler_len * 5)]
    for start in [1, sampler_len - 1, sampler_len, sampler_len + 5, sampler_len * 2 + 3]:
        r_stream = make_stream(start)
        resumed = [next(r_stream) for _ in range(5)]
        assert resumed == full[start : start + 5]


def test_normalize_prompt():
    assert normalize_prompt("  a \n  b\t c ") == "a b c"


def test_is_english():
    assert is_english("a photo of a cat")
    assert not is_english("猫の写真")


def test_is_clean():
    assert is_clean("a red sports car at sunset")
    assert not is_clean("a cat <lora:style>")
    assert not is_clean("a cat, trending on artstation")
    assert not is_clean("a nude woman")
