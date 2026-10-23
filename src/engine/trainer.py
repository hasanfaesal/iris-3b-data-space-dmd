"""Data-space DMD trainer: 4:1 TTUR, fake warmup, accumulation, EMA, sentinel.

Batch contract (Stage 5 / cached-embedding loader must yield this, one dict per
GLOBAL batch, all rows in one aspect bucket):

    {"y": [B, ...] text states, "y_mask": [B, L], "hw": (H, W)}

Rows are padded to a common length by the loader. The CFG null embedding is
constant, so it is passed once to ``Trainer`` ([1, ...] / [1, L]).

Schedule is a pure function of the counters (resume-safe): every turn consumes
one global batch; after ``warmup`` fake-only turns the pattern is
``fake_per_gen`` fake turns, then one generator turn.
"""

from __future__ import annotations

import contextlib
import errno
from dataclasses import dataclass
from pathlib import Path

import torch

from ..models.lora import use_adapter
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from .losses import fake_critic_loss, generator_loss
from .rollout import rollout_dataspace, sample_shifted_sigma


@dataclass
class TrainConfig:
    gen_lr: float = 2e-5
    fake_lr: float = 5e-5
    betas: tuple[float, float] = (0.0, 0.95)  # PLAN.md; turbo used (0, 0.999)
    weight_decay: float = 0.0
    clip: float = 1.0
    ema_decay: float = 0.999
    ema_slow_decay: float = 0.9999  # second, slower shadow (checkpointed; not previewed)
    fake_per_gen: int = 4
    warmup: int = 200  # fake-only turns before the first generator update
    micro_batch: int = 4  # global batch = len(batch) rows, split into micro-batches
    num_steps: int = 4  # R; micro-batch k uses r = k % num_steps
    shift: float = 4.0
    cfg_scale: float = 3.0
    cfg_window: float = 1.0  # shifted sigma' upper bound for CFG; 1.0 = every row
    preview_every: int | list[int] | set[int] = 50  # generator updates; 0 disables, or list of steps
    ckpt_every: int = 250  # slim gen+EMA file; 0 disables
    rolling_every: int = 100  # overwrite checkpoints/last.pt; 0 disables


class EMA:
    """EMA of the generator's trainable tensors, with a swap-in context manager."""

    def __init__(self, params: dict[str, torch.nn.Parameter], decay: float):
        self.params, self.decay = params, decay
        self.shadow = {n: p.detach().clone() for n, p in params.items()}

    @torch.no_grad()
    def update(self, step: int) -> None:
        # warmup ramp: zero-init weights would otherwise stay ~(1-decay^step) absorbed early on
        decay = min(self.decay, (1 + step) / (10 + step))
        for n, p in self.params.items():
            self.shadow[n].lerp_(p.detach(), 1 - decay)

    @contextlib.contextmanager
    def applied(self):
        backup = {n: p.detach().clone() for n, p in self.params.items()}
        with torch.no_grad():
            for n, p in self.params.items():
                p.copy_(self.shadow[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in self.params.items():
                    p.copy_(backup[n])


class StdSentinel:
    """Collapse guard: trips if pixel std collapses (< min_std) or explodes to static (> max_std)."""

    def __init__(
        self,
        min_std: float | None = None,
        max_std: float = 0.4,
        patience: int = 2,
        teacher_std: float | None = None,
        ratio: float = 0.6,
        collapse_after: int = 0,
    ):
        if min_std is None:
            min_std = ratio * teacher_std if teacher_std is not None else 0.2
        self.min_std, self.max_std, self.patience = min_std, max_std, patience
        self.teacher_std, self.collapse_after = teacher_std, collapse_after
        self.low = self.high = 0

    def update(self, std: float, step: int = 0) -> bool:
        # Static is never a draft. Collapse is: v4 was still ~0.19 at update 50.
        if std > self.max_std:
            self.high += 1
        else:
            self.high = 0
        if step >= self.collapse_after:
            self.low = self.low + 1 if std < self.min_std else 0
        self.bad = max(self.low, self.high)
        return self.bad >= self.patience


def _is_gen_param(name: str) -> bool:
    return ".adapters.gen." in name or name.startswith(("x0_embedder.", "step_embeddings."))


def _is_fake_param(name: str) -> bool:
    return ".adapters.fake." in name or name.startswith("fake_step_embeddings.")


class Trainer:
    def __init__(self, model, null, null_mask, cfg: TrainConfig, device="cpu", autocast_dtype=None, out_dir=None):
        self.model, self.cfg, self.device = model, cfg, torch.device(device)
        self.autocast_dtype = autocast_dtype
        self.out_dir = Path(out_dir) if out_dir else None
        self.null, self.null_mask = null.to(self.device), null_mask.to(self.device)

        named = dict(model.named_parameters())
        self.gen_params = {n: p for n, p in named.items() if _is_gen_param(n)}
        self.fake_params = {n: p for n, p in named.items() if _is_fake_param(n)}
        if not any(".adapters.gen." in n for n in self.gen_params) or not any(".adapters.fake." in n for n in self.fake_params):
            raise ValueError('inject_lora(model, "gen") and inject_lora(model, "fake") first')
        for n, p in named.items():  # base stays frozen
            p.requires_grad_(n in self.gen_params or n in self.fake_params)

        opt = dict(betas=cfg.betas, weight_decay=cfg.weight_decay)
        self.gen_opt = torch.optim.AdamW(self.gen_params.values(), lr=cfg.gen_lr, **opt)
        self.fake_opt = torch.optim.AdamW(self.fake_params.values(), lr=cfg.fake_lr, **opt)
        self.ema = EMA(self.gen_params, cfg.ema_decay)
        self.ema_slow = EMA(self.gen_params, cfg.ema_slow_decay)
        self.gen_updates = 0
        self.fake_updates = 0
        self._gen_up_params = [
            p for n, p in self.gen_params.items()
            if ".adapters.gen.1.weight" in n
        ]

    # -- schedule ------------------------------------------------------------------
    @property
    def turns(self) -> int:
        """Global batches consumed so far; a resumed loader must skip this many."""
        return self.gen_updates + self.fake_updates

    def next_is_gen(self) -> bool:
        c = self.cfg
        return self.fake_updates >= c.warmup + (self.gen_updates + 1) * c.fake_per_gen

    # -- updates -------------------------------------------------------------------
    def _autocast(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.autocast_dtype)

    def _micro_batches(self, batch):
        h, w = batch["hw"]
        rows = batch["y"].shape[0]
        for k, s in enumerate(range(0, rows, self.cfg.micro_batch)):
            y = batch["y"][s : s + self.cfg.micro_batch].to(self.device)
            mask = batch["y_mask"][s : s + self.cfg.micro_batch].to(self.device)
            b = y.shape[0]
            z = torch.randn(b, 3, h, w, device=self.device)
            noise = torch.randn_like(z)
            sigma = sample_shifted_sigma(b, self.cfg.shift, self.device)
            # offset by turns so every r is trained even when a batch has fewer micro-batches than steps
            yield (self.turns + k) % self.cfg.num_steps, y, mask, z, noise, sigma

    def _step(self, params, opt, adapter, loss_fn, batch) -> dict:
        chunks = list(self._micro_batches(batch))
        opt.zero_grad(set_to_none=True)
        total = 0.0
        extra_metrics = {}
        for r, y, mask, z, noise, sigma in chunks:
            # backward inside the adapter context: checkpoint recompute must see the live adapter
            with use_adapter(self.model, adapter):
                with self._autocast():
                    res = loss_fn(r, y, mask, z, noise, sigma)
                    if isinstance(res, tuple):
                        loss, aux = res
                        for k, v in aux.items():
                            extra_metrics[k] = extra_metrics.get(k, 0.0) + v / len(chunks)
                    else:
                        loss = res
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {adapter} loss at turn {self.turns}")
                (loss / len(chunks)).backward()
            total += loss.item() / len(chunks)
        grad_norm = torch.nn.utils.clip_grad_norm_(list(params.values()), self.cfg.clip)
        opt.step()
        out = {"loss": total, "grad_norm": grad_norm.item()}
        out.update(extra_metrics)
        return out

    def gen_update(self, batch) -> dict:
        def loss_fn(r, y, mask, z, noise, sigma):
            n = self.null.expand(y.shape[0], *self.null.shape[1:])
            nm = self.null_mask.expand(y.shape[0], *self.null_mask.shape[1:])
            c = self.cfg
            return generator_loss(self.model, z, y, mask, r, sigma, noise, n, nm, c.cfg_scale, c.cfg_window, return_aux=True)

        stats = self._step(self.gen_params, self.gen_opt, "gen", loss_fn, batch)
        self.ema.update(self.gen_updates)
        self.ema_slow.update(self.gen_updates)
        self.gen_updates += 1

        # Absolute lora_up norm across generator adapter layers
        if self._gen_up_params:
            stats["lora_up_norm"] = sum(p.detach().norm().item() for p in self._gen_up_params) / len(self._gen_up_params)

        return stats

    def fake_update(self, batch) -> dict:
        def loss_fn(r, y, mask, z, noise, sigma):
            return fake_critic_loss(self.model, z, y, mask, r, sigma, noise)

        stats = self._step(self.fake_params, self.fake_opt, "fake", loss_fn, batch)
        self.fake_updates += 1
        return stats

    def turn(self, batch) -> tuple[str, dict]:
        kind = "gen" if self.next_is_gen() else "fake"
        return kind, (self.gen_update if kind == "gen" else self.fake_update)(batch)

    # -- preview / sentinel ----------------------------------------------------------
    @torch.no_grad()
    def _render_batch(self, batch, seed: int) -> list[torch.Tensor]:
        r = self.cfg.num_steps - 1
        g = torch.Generator(device=self.device).manual_seed(seed)
        h, w = batch["hw"]
        num_rows = batch["y"].shape[0]
        steps_imgs = [[] for _ in range(r + 1)]
        for s in range(0, num_rows, self.cfg.micro_batch):
            y = batch["y"][s : s + self.cfg.micro_batch].to(self.device)
            mask = batch["y_mask"][s : s + self.cfg.micro_batch].to(self.device)
            z = torch.randn(y.shape[0], 3, h, w, device=self.device, generator=g)
            with self._autocast():
                all_steps = rollout_dataspace(self.model, z, y, mask, r, track_grad=False, return_all_steps=True)
                for step_idx in range(r + 1):
                    steps_imgs[step_idx].append(all_steps[step_idx + 1].float().cpu())
        return [((torch.cat(imgs) + 1) / 2).clamp(0, 1) for imgs in steps_imgs]

    @torch.no_grad()
    def render_teacher(self, batch, seed: int = 0, steps: int = 20, cfg_scale: float | None = None, shift: float = 3.0) -> float:
        """20-step teacher on the preview batch, one micro-batch at a time.

        cfg defaults to the training guidance. Returns mean per-image pixel std on [0, 1].
        """
        from PIL import Image
        from iris3b.flow.solver import FlowDPMSolver

        cfg_scale = self.cfg.cfg_scale if cfg_scale is None else cfg_scale
        h, w = batch["hw"]
        g = torch.Generator(device=self.device).manual_seed(seed)
        stds, imgs = [], []
        was_training = self.model.training
        self.model.eval()  # Iris checkpoints only while training; a 20-step graph OOMs
        try:
            y_all, mask_all = batch["y"], batch["y_mask"]
            for s in range(0, y_all.shape[0], self.cfg.micro_batch):
                y = y_all[s : s + self.cfg.micro_batch].to(self.device)
                mask = mask_all[s : s + self.cfg.micro_batch].to(self.device)
                b = y.shape[0]
                z = torch.randn(b, 3, h, w, device=self.device, generator=g).float()
                null = self.null.expand(b, -1, -1, -1)
                cfg_mask = torch.cat([self.null_mask.expand(b, -1), mask])

                def model_fn(x, t, cond, y=y, mask=mask, cfg_mask=cfg_mask):
                    m = mask if cond is y else cfg_mask
                    with self._autocast(), use_adapter(self.model, "off"):
                        return self.model(x, t, cond, y_mask=m).x.float()

                solver = FlowDPMSolver(model_fn, num_timesteps=1000, cfg_scale=cfg_scale, prediction="v")
                sample = solver.sample(
                    z, y, null if cfg_scale != 1.0 else None, steps=steps, order=2, shift=shift
                ).clamp(-1, 1)
                img = ((sample.float().cpu() + 1) / 2).clamp(0, 1)
                stds.append(img.flatten(1).std(1))
                imgs.append(img)
        finally:
            self.model.train(was_training)
        img = torch.cat(imgs)
        if self.out_dir:
            def dump() -> None:
                d_teacher = self.out_dir / "previews" / "teacher"
                d_teacher.mkdir(parents=True, exist_ok=True)
                for i, x in enumerate(img):
                    Image.fromarray((x.permute(1, 2, 0) * 255).round().byte().numpy()).save(d_teacher / f"{i:02d}.png")

            _skip_if_full(dump)
        return torch.cat(stds).mean().item()

    @torch.no_grad()
    def preview(self, batch, step: int, seed: int = 0) -> tuple[float, float]:
        """Render the fixed batch with both raw generator and EMA generator across all steps; dump PNGs.

        Returns (raw_pixel_std, ema_pixel_std) on the final refinement step.
        """
        from PIL import Image

        # 1. Raw generator weights
        steps_raw = self._render_batch(batch, seed)
        raw_std = steps_raw[-1].flatten(1).std(1).mean().item()

        # 2. EMA weights
        with self.ema.applied():
            steps_ema = self._render_batch(batch, seed)
        ema_std = steps_ema[-1].flatten(1).std(1).mean().item()

        if self.out_dir:
            def dump() -> None:
                d_raw = self.out_dir / "previews" / f"step{step:06d}_raw"
                d_ema = self.out_dir / "previews" / f"step{step:06d}_ema"
                d_raw.mkdir(parents=True, exist_ok=True)
                d_ema.mkdir(parents=True, exist_ok=True)
                for r_idx, x_step in enumerate(steps_raw):
                    for i, img in enumerate(x_step):
                        Image.fromarray((img.permute(1, 2, 0) * 255).round().byte().numpy()).save(
                            d_raw / f"{i:02d}_r{r_idx}.png"
                        )
                for r_idx, x_step in enumerate(steps_ema):
                    for i, img in enumerate(x_step):
                        Image.fromarray((img.permute(1, 2, 0) * 255).round().byte().numpy()).save(
                            d_ema / f"{i:02d}_r{r_idx}.png"
                        )

            _skip_if_full(dump)

        return raw_std, ema_std

    # -- checkpointing -----------------------------------------------------------------
    def state_dict(self) -> dict:
        cpu = lambda params: {n: p.detach().cpu() for n, p in params.items()}  # noqa: E731
        rng = {"cpu": torch.get_rng_state()}
        if self.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(self.device)
        return {
            "gen": cpu(self.gen_params), "fake": cpu(self.fake_params),
            "ema": {n: t.cpu() for n, t in self.ema.shadow.items()},
            "ema_slow": {n: t.cpu() for n, t in self.ema_slow.shadow.items()},
            "gen_opt": self.gen_opt.state_dict(), "fake_opt": self.fake_opt.state_dict(),
            "gen_updates": self.gen_updates, "fake_updates": self.fake_updates, "rng": rng,
        }

    def load_state_dict(self, state: dict) -> None:
        with torch.no_grad():
            for n, p in self.gen_params.items():
                p.copy_(state["gen"][n])
            for n, p in self.fake_params.items():
                p.copy_(state["fake"][n])
            for n, t in self.ema.shadow.items():
                t.copy_(state["ema"][n])
            for n, t in self.ema_slow.shadow.items():
                t.copy_(state.get("ema_slow", state["ema"])[n])  # older checkpoints: seed from the fast EMA
        self.gen_opt.load_state_dict(state["gen_opt"])
        self.fake_opt.load_state_dict(state["fake_opt"])
        self.gen_updates, self.fake_updates = state["gen_updates"], state["fake_updates"]
        torch.set_rng_state(state["rng"]["cpu"])
        if "cuda" in state["rng"]:
            torch.cuda.set_rng_state(state["rng"]["cuda"], self.device)

    def state_dict_slim(self) -> dict:
        """Generator weights plus the fast EMA. The slow EMA and optimizer stay in ``last.pt``."""
        cpu = lambda params: {n: p.detach().cpu() for n, p in params.items()}  # noqa: E731
        return {
            "gen": cpu(self.gen_params),
            "ema": {n: t.cpu() for n, t in self.ema.shadow.items()},
            "gen_updates": self.gen_updates,
        }

    def save(self, path) -> bool:
        return save_checkpoint(self.state_dict(), path)

    def save_slim(self, path) -> bool:
        return save_checkpoint(self.state_dict_slim(), path)

    def load(self, path) -> None:
        state = load_checkpoint(path)
        if "gen_opt" not in state or "fake" not in state:
            raise ValueError(f"{path} has no optimizer state; resume from checkpoints/last.pt")
        self.load_state_dict(state)

    # -- loop ----------------------------------------------------------------------------
    def fit(
        self,
        batches,
        total_gen_updates: int,
        fixed_batch=None,
        sentinel: StdSentinel | None = None,
        platform_sentinel=None,
        log=print,
    ):
        """Run until ``total_gen_updates``; ``batches`` yields global batches (already
        advanced past ``self.turns`` when resuming). Returns True unless the sentinel stopped it."""
        c = self.cfg
        for batch in batches:
            if self.gen_updates >= total_gen_updates:
                break
            if platform_sentinel and (stop_reason := platform_sentinel.check()):
                log(f"platform stop: {stop_reason}. Saving checkpoint and exiting cleanly.")
                self._save_full()
                return True
            kind, stats = self.turn(batch)
            msg = (
                f"turn {self.turns} {kind} gen={self.gen_updates} fake={self.fake_updates} "
                f"loss={stats['loss']:.4g} gnorm={stats['grad_norm']:.3g}"
            )
            if "v_diff" in stats:
                msg += f" v_diff={stats['v_diff']:.4g}"
            if "lora_up_norm" in stats:
                msg += f" up_norm={stats['lora_up_norm']:.4f}"
            log(msg)
            if kind != "gen":
                continue
            should_preview = (
                (isinstance(c.preview_every, (list, tuple, set)) and self.gen_updates in c.preview_every)
                or (isinstance(c.preview_every, int) and c.preview_every > 0 and self.gen_updates % c.preview_every == 0)
            )
            if should_preview and fixed_batch is not None:
                raw_std, ema_std = self.preview(fixed_batch, self.gen_updates)
                log(f"preview gen={self.gen_updates} raw_pixel_std={raw_std:.4f} ema_pixel_std={ema_std:.4f}")
                if sentinel and sentinel.update(raw_std, self.gen_updates):
                    log(f"sentinel: raw pixel std ({raw_std:.4f}) outside bounds [{sentinel.min_std}, {sentinel.max_std}], stopping")
                    self._save_full()
                    if c.ckpt_every and self.gen_updates % c.ckpt_every == 0:
                        self._save_slim()
                    return False
                if sentinel and self.gen_updates < sentinel.collapse_after and raw_std < sentinel.min_std:
                    log(
                        f"sentinel: raw pixel std {raw_std:.4f} < {sentinel.min_std:.4f}, "
                        f"collapse check arms at gen {sentinel.collapse_after}"
                    )
            if c.rolling_every and self.gen_updates % c.rolling_every == 0:
                self._save_full()
            if c.ckpt_every and self.gen_updates % c.ckpt_every == 0:
                self._save_slim()
        return True

    def _save_full(self) -> None:
        if not self.out_dir:
            return
        path = self.out_dir / "checkpoints" / "last.pt"
        if self.save(path):
            print(f"full checkpoint {path}", flush=True)

    def _save_slim(self) -> None:
        if not self.out_dir:
            return
        path = self.out_dir / "checkpoints" / f"gen{self.gen_updates:06d}.pt"
        if self.save_slim(path):
            print(f"slim checkpoint {path}", flush=True)


def _skip_if_full(fn) -> None:
    try:
        fn()
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        print("write skipped: disk full", flush=True)
