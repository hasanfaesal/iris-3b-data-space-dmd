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
    fake_per_gen: int = 4
    warmup: int = 200  # fake-only turns before the first generator update
    micro_batch: int = 4  # global batch = len(batch) rows, split into micro-batches
    num_steps: int = 4  # R; micro-batch k uses r = k % num_steps
    shift: float = 4.0
    cfg_scale: float = 3.0
    cfg_window: float = 0.6
    preview_every: int = 250  # generator updates; 0 disables
    ckpt_every: int = 250  # generator updates; 0 disables


class EMA:
    """EMA of the generator's trainable tensors, with a swap-in context manager."""

    def __init__(self, params: dict[str, torch.nn.Parameter], decay: float):
        self.params, self.decay = params, decay
        self.shadow = {n: p.detach().clone() for n, p in params.items()}

    @torch.no_grad()
    def update(self) -> None:
        for n, p in self.params.items():
            self.shadow[n].lerp_(p.detach(), 1 - self.decay)

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
    """Collapse guard: trips after ``patience`` consecutive previews below ``min_std``.

    Pass min_std = 0.7 * teacher's per-image pixel std (PLAN.md). Turbo's
    absolute rule of thumb is 0.2 on [0,1] pixels.
    """

    def __init__(self, min_std: float = 0.2, patience: int = 2):
        self.min_std, self.patience, self.bad = min_std, patience, 0

    def update(self, std: float) -> bool:
        self.bad = self.bad + 1 if std < self.min_std else 0
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
        self.gen_updates = 0
        self.fake_updates = 0

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
            yield k % self.cfg.num_steps, y, mask, z, noise, sigma

    def _step(self, params, opt, adapter, loss_fn, batch) -> dict:
        chunks = list(self._micro_batches(batch))
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for r, y, mask, z, noise, sigma in chunks:
            # backward inside the adapter context: checkpoint recompute must see the live adapter
            with use_adapter(self.model, adapter):
                with self._autocast():
                    loss = loss_fn(r, y, mask, z, noise, sigma)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {adapter} loss at turn {self.turns}")
                (loss / len(chunks)).backward()
            total += loss.item() / len(chunks)
        grad_norm = torch.nn.utils.clip_grad_norm_(list(params.values()), self.cfg.clip)
        opt.step()
        return {"loss": total, "grad_norm": grad_norm.item()}

    def gen_update(self, batch) -> dict:
        def loss_fn(r, y, mask, z, noise, sigma):
            n = self.null.expand(y.shape[0], *self.null.shape[1:])
            nm = self.null_mask.expand(y.shape[0], *self.null_mask.shape[1:])
            c = self.cfg
            return generator_loss(self.model, z, y, mask, r, sigma, noise, n, nm, c.cfg_scale, c.cfg_window)

        stats = self._step(self.gen_params, self.gen_opt, "gen", loss_fn, batch)
        self.ema.update()
        self.gen_updates += 1
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
    def preview(self, batch, step: int, seed: int = 0) -> float:
        """Render the fixed batch with the EMA generator at the last step; dump PNGs.

        Returns mean per-image pixel std, assuming model pixels in [-1, 1]
        (mapped to [0, 1]). RNG state is untouched, so previews never perturb training.
        """
        from PIL import Image

        r = self.cfg.num_steps - 1
        g = torch.Generator(device=self.device).manual_seed(seed)
        h, w = batch["hw"]
        images = []
        with self.ema.applied():
            for s in range(0, batch["y"].shape[0], self.cfg.micro_batch):
                y = batch["y"][s : s + self.cfg.micro_batch].to(self.device)
                mask = batch["y_mask"][s : s + self.cfg.micro_batch].to(self.device)
                z = torch.randn(y.shape[0], 3, h, w, device=self.device, generator=g)
                with self._autocast():
                    images.append(rollout_dataspace(self.model, z, y, mask, r, track_grad=False).float().cpu())
        x = ((torch.cat(images) + 1) / 2).clamp(0, 1)
        if self.out_dir:
            d = self.out_dir / "previews" / f"step{step:06d}"
            d.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(x):
                Image.fromarray((img.permute(1, 2, 0) * 255).round().byte().numpy()).save(d / f"{i:02d}.png")
        return x.flatten(1).std(1).mean().item()

    # -- checkpointing -----------------------------------------------------------------
    def state_dict(self) -> dict:
        cpu = lambda params: {n: p.detach().cpu() for n, p in params.items()}  # noqa: E731
        rng = {"cpu": torch.get_rng_state()}
        if self.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(self.device)
        return {
            "gen": cpu(self.gen_params), "fake": cpu(self.fake_params),
            "ema": {n: t.cpu() for n, t in self.ema.shadow.items()},
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
        self.gen_opt.load_state_dict(state["gen_opt"])
        self.fake_opt.load_state_dict(state["fake_opt"])
        self.gen_updates, self.fake_updates = state["gen_updates"], state["fake_updates"]
        torch.set_rng_state(state["rng"]["cpu"])
        if "cuda" in state["rng"]:
            torch.cuda.set_rng_state(state["rng"]["cuda"], self.device)

    def save(self, path) -> None:
        save_checkpoint(self.state_dict(), path)

    def load(self, path) -> None:
        self.load_state_dict(load_checkpoint(path))

    # -- loop ----------------------------------------------------------------------------
    def fit(self, batches, total_gen_updates: int, fixed_batch=None, sentinel: StdSentinel | None = None, log=print):
        """Run until ``total_gen_updates``; ``batches`` yields global batches (already
        advanced past ``self.turns`` when resuming). Returns True unless the sentinel stopped it."""
        c = self.cfg
        for batch in batches:
            if self.gen_updates >= total_gen_updates:
                break
            kind, stats = self.turn(batch)
            log(f"turn {self.turns} {kind} gen={self.gen_updates} fake={self.fake_updates} "
                f"loss={stats['loss']:.4g} gnorm={stats['grad_norm']:.3g}")
            if kind != "gen":
                continue
            if c.preview_every and fixed_batch is not None and self.gen_updates % c.preview_every == 0:
                std = self.preview(fixed_batch, self.gen_updates)
                log(f"preview gen={self.gen_updates} pixel_std={std:.4f}")
                if sentinel and sentinel.update(std):
                    log("sentinel: pixel std collapsed, stopping")
                    self._save_ckpt()
                    return False
            if c.ckpt_every and self.gen_updates % c.ckpt_every == 0:
                self._save_ckpt()
        return True

    def _save_ckpt(self) -> None:
        if self.out_dir:
            self.save(self.out_dir / "checkpoints" / f"gen{self.gen_updates:06d}.pt")
