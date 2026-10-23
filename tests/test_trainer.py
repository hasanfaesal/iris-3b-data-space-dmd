"""Stage 6 checks: 20-turn CPU run, schedule, live adapters, exact resume, sentinel."""

import importlib.util
import itertools
import json
from dataclasses import replace
from pathlib import Path

import torch

from src.engine.trainer import EMA, StdSentinel, TrainConfig, Trainer, rollback_gen_lr
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
    return Trainer(model, null, null_mask, replace(CFG), out_dir=out_dir)


def test_20_turns_finite_schedule_and_live_adapters():
    tr = make_trainer()
    kinds = []
    for t in range(20):
        kind, stats = tr.turn(batch_at(t))
        kinds.append(kind)
        assert all(map(torch.isfinite, (torch.tensor(stats["loss"]), torch.tensor(stats["grad_norm"]))))
        if kind == "gen":
            assert "lora_up_norm" in stats and stats["lora_up_norm"] > 0
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


def _train_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preview_rows_match_eval_file(tmp_path):
    mod = _train_script()
    prompts = Path(__file__).resolve().parents[1] / "data" / "prompts_eval.jsonl"
    rows = [json.loads(line)["prompt"] for line in prompts.read_text().splitlines() if line.strip()]
    for i, text in mod.PREVIEW_ROWS:
        assert rows[i] == text and i < 100

    class Cache:
        dir = tmp_path
        n = 200

        def get(self, indices):
            self.got = list(indices)
            return torch.zeros(len(indices), 1), torch.zeros(len(indices))

    (tmp_path / "index.json").write_text(json.dumps({"sha256": mod._sha256(prompts), "n": 200}))
    batch = mod.load_preview_batch(Cache(), prompts)
    assert batch["hw"] == (1024, 1024) and batch["y"].shape[0] == len(mod.PREVIEW_ROWS)
    (tmp_path / "index.json").write_text(json.dumps({"sha256": "0", "n": 200}))
    try:
        mod.load_preview_batch(Cache(), prompts)
    except SystemExit:
        pass
    else:
        raise AssertionError("stale eval cache should abort")


def test_sentinel_ignores_draft_collapse_and_still_catches_static():
    sentinel = StdSentinel(teacher_std=0.30, ratio=0.6, patience=2, max_std=0.40, collapse_after=100)
    assert not sentinel.update(0.10, 50) and sentinel.low == 0
    assert not sentinel.update(0.10, 100)
    assert sentinel.update(0.10, 150)
    static = StdSentinel(teacher_std=0.30, ratio=0.6, patience=2, max_std=0.40, collapse_after=100)
    assert not static.update(0.50, 50)
    assert static.update(0.50, 100)


def test_sentinel_teacher_ratio_and_patience():
    # 0.6 * teacher_std (0.30 -> 0.18)
    sentinel = StdSentinel(teacher_std=0.30, ratio=0.6, patience=2, max_std=0.40)
    assert abs(sentinel.min_std - 0.18) < 1e-6
    # One low preview should not stop (patience=2)
    assert not sentinel.update(0.17)
    # Recovering back above threshold resets counter
    assert not sentinel.update(0.22)
    # One low preview again
    assert not sentinel.update(0.17)
    # Two low previews in a row trips
    assert sentinel.update(0.17)
    # Static explosion check (> max_std)
    sentinel_static = StdSentinel(teacher_std=0.30, ratio=0.6, patience=2, max_std=0.40)
    assert not sentinel_static.update(0.42)
    assert sentinel_static.update(0.45)


def test_fit_preview_sentinel_and_checkpoints(tmp_path):
    tr = make_trainer(tmp_path)
    tr.cfg.preview_every, tr.cfg.ckpt_every = 1, 1
    sentinel = StdSentinel(min_std=10.0, patience=2)  # impossible to satisfy -> trips at 2nd preview
    done = tr.fit(batches(), total_gen_updates=5, fixed_batch=batch_at(99), sentinel=sentinel, log=lambda _: None)
    assert done is False and tr.gen_updates == 2
    assert len(list((tmp_path / "previews").rglob("*.png"))) == 32  # 4 rows x 2 previews x 2 steps x 2 (raw + ema)
    full = tmp_path / "checkpoints" / "last.pt"
    stopped = tmp_path / "checkpoints" / "gen000002.pt"
    assert full.is_file() and stopped.is_file()
    state = torch.load(full, map_location="cpu", weights_only=False)
    assert "gen_opt" in state and "fake" in state
    assert "gen_opt" not in torch.load(stopped, map_location="cpu", weights_only=False)


def test_every_r_trained_when_batch_has_fewer_microbatches_than_steps():
    tr = make_trainer()
    tr.cfg = TrainConfig(micro_batch=4, num_steps=4)  # 4 rows -> ONE micro-batch, 4 steps
    seen = set()
    for _ in range(4):
        seen |= {r for r, *_ in tr._micro_batches(batch_at(0))}
        tr.fake_updates += 1
    assert seen == {0, 1, 2, 3}


def test_one_rolling_full_and_slim_without_optimizer(tmp_path):
    tr = make_trainer(tmp_path)
    tr.cfg = replace(tr.cfg, rolling_every=1, ckpt_every=2, preview_every=0)
    assert tr.fit(batches(), total_gen_updates=2, log=lambda _: None)
    ckpt = tmp_path / "checkpoints"
    assert {p.name for p in ckpt.glob("*.pt")} == {"last.pt", "gen000002.pt"}
    full = torch.load(ckpt / "last.pt", map_location="cpu", weights_only=False)
    slim = torch.load(ckpt / "gen000002.pt", map_location="cpu", weights_only=False)
    assert "gen_opt" in full and "fake_opt" in full and "ema_slow" in full
    assert set(slim) == {"gen", "ema", "gen_updates"} and slim["gen_updates"] == 2
    other = make_trainer()
    try:
        other.load(ckpt / "gen000002.pt")
    except ValueError as e:
        assert "last.pt" in str(e)
    else:
        raise AssertionError("slim checkpoint must not resume")


def test_checkpoint_skips_when_disk_is_full(tmp_path, monkeypatch):
    import errno

    import src.utils.checkpoint as ckpt

    nested = {"a": torch.zeros(4), "b": {"c": torch.zeros(4)}}
    assert ckpt.tensor_bytes(nested) == 32

    monkeypatch.setattr(ckpt.shutil, "disk_usage", lambda _p: type("U", (), {"free": 0})())
    assert ckpt.save_checkpoint({"w": torch.zeros(8)}, tmp_path / "a.pt") is False
    assert not (tmp_path / "a.pt").exists()

    monkeypatch.setattr(ckpt.shutil, "disk_usage", lambda _p: type("U", (), {"free": 1 << 40})())

    def boom(_state, path):
        Path(path).write_bytes(b"partial")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(ckpt.torch, "save", boom)
    assert ckpt.save_checkpoint({"w": torch.zeros(4)}, tmp_path / "b.pt") is False
    assert not (tmp_path / "b.pt").exists() and not (tmp_path / "b.pt.tmp").exists()


def test_drop_diagnostic_runs_keeps_the_current_out(tmp_path):
    mod = _train_script()
    runs = tmp_path / "runs"
    (runs / "diagnostic_v4" / "checkpoints").mkdir(parents=True)
    (runs / "diagnostic_v4" / "checkpoints" / "last.pt").write_bytes(b"x")
    (runs / "main").mkdir()
    (runs / "keepme").mkdir()
    mod.drop_diagnostic_runs(runs, runs / "main")
    assert not (runs / "diagnostic_v4").exists()
    assert (runs / "main").is_dir() and (runs / "keepme").is_dir()
    (runs / "diagnostic_v5").mkdir()
    mod.drop_diagnostic_runs(runs, runs / "diagnostic_v5")
    assert (runs / "diagnostic_v5").is_dir()


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


def test_platform_sentinel_deadline_and_clean_exit(tmp_path):
    from src.utils.platform import PlatformSentinel, parse_deadline

    # Test parser
    assert parse_deadline(None) is None
    ts = parse_deadline("2026-10-26T06:53:00+00:00")
    assert ts is not None and parse_deadline("2026-10-26 06:53:00 UTC") == ts

    # Triggered deadline
    ps = PlatformSentinel(deadline=100.0, check_interval=0)
    assert ps.check(now=101.0) is not None

    # Untriggered deadline
    ps = PlatformSentinel(deadline=200.0, check_interval=0)
    assert ps.check(now=101.0) is None

    # Trainer fit clean exit on platform sentinel
    tr = make_trainer(tmp_path)
    tr.cfg.ckpt_every = 100
    mock_sentinel = PlatformSentinel(deadline=50.0, check_interval=0)
    # mock_sentinel check will immediately trigger at now=100.0
    mock_sentinel.check = lambda: "test deadline reached"
    ok = tr.fit(batches(), total_gen_updates=10, platform_sentinel=mock_sentinel, log=lambda _: None)
    assert ok is True  # platform early stop is clean exit
    assert (tmp_path / "checkpoints" / "last.pt").is_file()


def test_rollback_lr_never_raises():
    assert rollback_gen_lr("static", 2e-5) == 5e-6
    assert rollback_gen_lr("collapse", 2e-5) == 1e-5
    assert rollback_gen_lr("nan", 2e-5) == 1e-5
    assert rollback_gen_lr("collapse", 1e-6) == 1e-6


def test_rollback_once_then_stop(tmp_path, monkeypatch):
    tr = make_trainer(tmp_path)
    tr.cfg.preview_every, tr.cfg.rolling_every, tr.cfg.ckpt_every = 1, 0, 0
    seen = []

    def fake_preview(batch, step, seed=0):
        std = 0.25 if len(seen) < 2 else 0.50  # two healthy previews, then static
        seen.append(step)
        return std, std

    monkeypatch.setattr(tr, "preview", fake_preview)
    sentinel = StdSentinel(min_std=0.18, max_std=0.40, patience=2)
    done = tr.fit(batches(), total_gen_updates=20, fixed_batch=batch_at(99), sentinel=sentinel, log=lambda _: None)
    assert done is False and tr.rolled_back
    assert tr.gen_opt.param_groups[0]["lr"] == 5e-6
    assert tr.fake_opt.param_groups[0]["lr"] == 5e-5  # critic LR is not touched
    good = torch.load(tmp_path / "checkpoints" / "last_good.pt", map_location="cpu", weights_only=False)
    last = torch.load(tmp_path / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    assert good["gen_updates"] == last["gen_updates"] == 2  # the second stop did not overwrite it
    assert good["rolled_back"] and good["gen_opt"]["param_groups"][0]["lr"] == 5e-6
    assert good["fake_opt"]["param_groups"][0]["lr"] == 5e-5
    kinds = [json.loads(ln)["kind"] for ln in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert kinds.count("rollback") == 1 and kinds[-1] == "stop"


def test_nan_rolls_back_to_collapse_lr(tmp_path, monkeypatch):
    tr = make_trainer(tmp_path)
    tr.cfg.preview_every, tr.cfg.rolling_every, tr.cfg.ckpt_every = 1, 0, 0
    monkeypatch.setattr(tr, "preview", lambda batch, step, seed=0: (0.25, 0.25))
    real_turn = tr.turn

    def exploding(batch):  # the turn after the first generator update raises, and so does the one after the restore
        if tr.gen_updates >= 1:
            raise FloatingPointError("boom")
        return real_turn(batch)

    monkeypatch.setattr(tr, "turn", exploding)
    sentinel = StdSentinel(min_std=0.05, max_std=0.90, patience=2)
    done = tr.fit(batches(), total_gen_updates=20, fixed_batch=batch_at(99), sentinel=sentinel, log=lambda _: None)
    assert done is False and tr.rolled_back
    assert tr.gen_opt.param_groups[0]["lr"] == 1e-5
    assert tr.fake_opt.param_groups[0]["lr"] == 5e-5


def test_fit_writes_metrics_and_status(tmp_path):
    tr = make_trainer(out_dir=tmp_path)
    ok = tr.fit(batches(), total_gen_updates=2, log=lambda _: None)
    assert ok
    lines = [json.loads(ln) for ln in (tmp_path / "metrics.jsonl").read_text().splitlines() if ln.strip()]
    assert lines[-1]["kind"] == "turn"
    assert lines[-1]["gen_updates"] == 2
    assert all(ln["kind"] == "turn" for ln in lines)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["health"] == "complete"
    assert status["gen_updates"] == 2
    assert status["target_gen_updates"] == 2
