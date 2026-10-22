"""Stage 6 checks: 20-turn CPU run, schedule, live adapters, exact resume, sentinel."""

import itertools

import torch

from src.engine.trainer import EMA, StdSentinel, TrainConfig, Trainer
from src.models.lora import inject_lora, lora_layers
from tiny import dataspace_iris, tiny_cond

CFG = TrainConfig(warmup=2, fake_per_gen=2, micro_batch=2, num_steps=2, preview_every=0, ckpt_every=0, ema_decay=0.9)


def batch_at(turn: int, rows: int = 4):
    """Deterministic batch as a function of the turn index (stands in for the cached-embedding loader)."""
    torch.manual_seed(1000 + turn)
    y, mask = tiny_cond(rows)
    return {"y": y, "y_mask": mask, "hw": (16, 16)}


def batches(start: int = 0):
    return (batch_at(t) for t in itertools.count(start))


def make_trainer(out_dir=None):
    torch.manual_seed(0)
    model = dataspace_iris()
    inject_lora(model, "gen", 8)
    inject_lora(model, "fake", 4)
    null, null_mask = tiny_cond(1)
    return Trainer(model, null, null_mask, CFG, out_dir=out_dir)


def test_20_turns_finite_schedule_and_live_adapters():
    tr = make_trainer()
    kinds = []
    for t in range(20):
        kind, stats = tr.turn(batch_at(t))
        kinds.append(kind)
        assert all(map(torch.isfinite, (torch.tensor(stats["loss"]), torch.tensor(stats["grad_norm"]))))
    # warmup 2 fake, then (2 fake, 1 gen) repeating
    assert kinds[:8] == ["fake"] * 4 + ["gen"] + ["fake"] * 2 + ["gen"]
    assert tr.turns == 20 and tr.gen_updates == kinds.count("gen") and tr.fake_updates == kinds.count("fake")
    for name in ("gen", "fake"):  # dead-adapter check: every B moved off zero
        dead = [n for n, m in lora_layers(tr.model).items() if m.adapters[name][1].weight.abs().sum() == 0]
        assert not dead, f"{name} adapters never trained: {dead[:5]}"
    assert tr.model.x0_embedder.weight.abs().sum() > 0
    assert tr.model.step_embeddings.weight.abs().sum() > 0 and tr.model.fake_step_embeddings.weight.abs().sum() > 0
    assert all(not p.requires_grad for n, p in tr.model.named_parameters() if n not in tr.gen_params | tr.fake_params)


def test_save_resume_reproduces_next_steps_exactly(tmp_path):
    ref = make_trainer()
    for t in range(10):
        ref.turn(batch_at(t))

    a = make_trainer()
    for t in range(5):
        a.turn(batch_at(t))
    a.save(tmp_path / "ckpt.pt")

    b = make_trainer()
    b.load(tmp_path / "ckpt.pt")
    assert b.turns == 5
    for t in range(b.turns, 10):  # a resumed loader skips `turns` batches
        b.turn(batch_at(t))

    sa, sb = ref.state_dict(), b.state_dict()
    for group in ("gen", "fake", "ema", "ema_slow"):
        assert all(torch.equal(sa[group][n], sb[group][n]) for n in sa[group]), group


def test_fit_preview_sentinel_and_checkpoints(tmp_path):
    tr = make_trainer(tmp_path)
    tr.cfg.preview_every, tr.cfg.ckpt_every = 1, 1
    sentinel = StdSentinel(min_std=10.0, patience=2)  # impossible to satisfy -> trips at 2nd preview
    done = tr.fit(batches(), total_gen_updates=5, fixed_batch=batch_at(99), sentinel=sentinel, log=lambda _: None)
    assert done is False and tr.gen_updates == 2
    assert len(list((tmp_path / "previews").rglob("*.png"))) == 8  # 4 rows x 2 previews
    assert (tmp_path / "checkpoints" / "gen000002.pt").exists()


def test_every_r_trained_when_batch_has_fewer_microbatches_than_steps():
    tr = make_trainer()
    tr.cfg = TrainConfig(micro_batch=4, num_steps=4)  # 4 rows -> ONE micro-batch, 4 steps
    seen = set()
    for _ in range(4):
        seen |= {r for r, *_ in tr._micro_batches(batch_at(0))}
        tr.fake_updates += 1
    assert seen == {0, 1, 2, 3}


def test_ema_warmup_tracks_early_updates():
    """Step 0 copies the live weight (decay 0.1 at most); late steps cap at cfg decay."""
    p = {"w": torch.nn.Parameter(torch.zeros(1))}
    ema = EMA(p, 0.999)
    p["w"].data.fill_(1.0)
    ema.update(0)
    assert ema.shadow["w"].item() > 0.85  # fixed 0.999 would give 0.001
    ema.shadow["w"].zero_()
    ema.update(10**6)
    assert abs(ema.shadow["w"].item() - 0.001) < 1e-6
