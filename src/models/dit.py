"""Data-space iteration wrapper around IrisDiT (arXiv:2610.12102).

Subclasses the vendored IrisDiT instead of copying its forward. The two new
branches are injected with forward hooks on ``s_embedder`` and ``t_embedder``,
which fire inside ``super().forward()``; the hooks are no-ops unless a stash is
armed. Therefore:

- role ``"off"`` runs ``super().forward`` untouched -> teacher parity by
  construction (branch weights are never read);
- role ``"gen"`` adds the x0 patch branch and the step embedding;
- role ``"fake"`` adds only its own step embedding (its own table, so the fake
  loss cannot move the generator's embeddings).

All new parameters are zero-initialized: at init, "gen" with any x0 and "fake"
are bit-identical to the base teacher at the same t.

Meta-device construction order (H100 scripts):
    with torch.device("meta"): model = DataSpaceIris(cfg.model)
    missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    assert set(missing) == set(DataSpaceIris.BRANCH_KEYS) and not unexpected  # base ckpt lacks the branches
    for m in (model.x0_embedder, model.step_embeddings, model.fake_step_embeddings):
        m.to_empty(device=...)
    model.reset_dataspace_init()
(see src/models/loading.py)

The new parameters (x0_embedder.weight, step_embeddings.weight,
fake_step_embeddings.weight) are NOT LoRA adapter slots: the trainer adds them
to the gen / fake optimizer param groups explicitly.
"""

import torch
import torch.nn.functional as F
from torch import nn

from iris3b.config import ModelConfig
from iris3b.models.dit import IrisDiT, IrisOutput

from .lora import lora_layers


class DataSpaceIris(IrisDiT):
    # state_dict keys owned by this class; base checkpoints must be loaded with
    # strict=False and these are the only allowed unexpected keys
    BRANCH_KEYS = ("x0_embedder.weight", "step_embeddings.weight", "fake_step_embeddings.weight")

    def __init__(self, cfg: ModelConfig, num_steps: int = 4):
        super().__init__(cfg)
        dim = cfg.hidden_size
        self.x0_embedder = nn.Linear(cfg.patch_size * cfg.patch_size * cfg.in_channels, dim, bias=False)
        self.step_embeddings = nn.Embedding(num_steps, dim)  # generator's r table
        self.fake_step_embeddings = nn.Embedding(num_steps, dim)  # fake critic's own r table
        self.reset_dataspace_init()
        self._dataspace_stash: dict | None = None  # per-call branch deltas; not a submodule
        self.s_embedder.register_forward_hook(self._s_embedder_hook)
        self.t_embedder.register_forward_hook(self._t_embedder_hook)

    def reset_dataspace_init(self) -> None:
        """Zero the new branches (call again after to_empty on a meta build)."""
        nn.init.zeros_(self.x0_embedder.weight)
        nn.init.zeros_(self.step_embeddings.weight)
        nn.init.zeros_(self.fake_step_embeddings.weight)

    # -- branch hooks ----------------------------------------------------------
    def _s_embedder_hook(self, _module, _inputs, output):
        stash = self._dataspace_stash
        if stash is None or "x0_patches" not in stash:
            return output
        return output + self.x0_embedder(stash["x0_patches"])

    def _t_embedder_hook(self, _module, _inputs, output):
        stash = self._dataspace_stash
        if stash is None or "step_emb" not in stash:
            return output
        return output + stash["step_emb"]

    # -- role plumbing -----------------------------------------------------------
    def _active_role(self) -> str:
        """Read the role from the LoRA layers' live state (single source of truth)."""
        actives = {layer.active for layer in lora_layers(self).values()}
        if not actives:
            return "off"  # adapters not injected: plain base
        if len(actives) > 1:
            raise RuntimeError(f"mixed adapter state {actives}: use_adapter must wrap the whole forward")
        active = actives.pop()
        return "off" if active is None else active

    # -- forward -----------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        x0: torch.Tensor | None = None,
        r: torch.Tensor | int | None = None,
        capture: tuple[int, ...] = (),
        y_mask: torch.Tensor | None = None,
    ) -> IrisOutput:
        """Data-space forward.

        Args follow IrisDiT.forward (x, t, y, capture, y_mask) plus:
            x0: current data estimate [B, C, H, W] -- "gen" only.
            r: rollout step index, int or [B] long -- required for "gen"/"fake".
        """
        role = self._active_role()
        if role == "off":
            if x0 is not None or r is not None:
                raise ValueError('x0/r are generator/fake inputs; the teacher role ("off") takes none')
            return super().forward(x, t, y, capture=capture, y_mask=y_mask)

        if role == "gen":
            if x0 is None:
                raise ValueError("role 'gen' requires x0 (the current data estimate)")
            if x0.shape != x.shape:
                raise ValueError(f"x0 shape {tuple(x0.shape)} != x shape {tuple(x.shape)}")
        elif x0 is not None:
            raise ValueError("role 'fake' takes noisy x only; it has no x0 branch")
        if r is None:
            raise ValueError(f"role '{role}' requires r (the rollout step index)")

        batch = x.shape[0]
        if not torch.is_tensor(r):
            r = torch.full((batch,), r, device=x.device, dtype=torch.long)
        r = r.to(device=x.device, dtype=torch.long)
        table = self.step_embeddings if role == "gen" else self.fake_step_embeddings
        step_emb = table(r).view(batch, 1, -1)

        stash: dict = {"step_emb": step_emb}
        if role == "gen":
            p = self.cfg.patch_size
            stash["x0_patches"] = F.unfold(x0.to(x.dtype), kernel_size=p, stride=p).transpose(1, 2)
        self._dataspace_stash = stash
        try:
            return super().forward(x, t, y, capture=capture, y_mask=y_mask)
        finally:
            self._dataspace_stash = None