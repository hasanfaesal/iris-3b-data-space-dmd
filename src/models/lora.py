"""Role-swapped LoRA on a frozen base IrisDiT (ported from pentaCoxian/iris-3b-turbo).

Several named adapters share one frozen base model; ``use_adapter(model, name)``
selects which one is live. Our roles:

- ``"off"`` (or ``None``): the bare base weights — the frozen teacher, bit-identical;
- ``"gen"``: the data-space generator adapter;
- ``"fake"``: the DMD fake-critic adapter.

A fresh adapter is an exact no-op (zero-initialized ``up`` projection), so at
init the generator equals the base at t=1000. For inference an adapter file is
folded into the plain weights (``merge_lora_file``).

Note: ``rebind_shared_cores`` lives here because the official iris-3b package
does not have it (the turbo fork does): shared adaLN cores are referenced by
every block through an unregistered ``_core`` tuple that must be pointed at the
wrapped linear.
"""

import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

# submodules of IrisDiT that receive adapters; the text adapter stays frozen
# because the prompt states it produces are shared by every timestep
DEFAULT_TARGETS = ("s_embedder", "t_embedder", "modulation_cores", "blocks", "pixel_blocks", "final_layer")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self.scales: dict[str, float] = {}
        self.active: str | None = None

    def add_adapter(self, name: str, rank: int, alpha: float) -> None:
        down = nn.Linear(self.base.in_features, rank, bias=False)
        up = nn.Linear(rank, self.base.out_features, bias=False)
        nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
        nn.init.zeros_(up.weight)  # a fresh adapter is an exact no-op
        device = self.base.weight.device
        self.adapters[name] = nn.Sequential(down, up).to(device=device, dtype=torch.float32)
        self.scales[name] = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active is None:
            return out
        return out + self.adapters[self.active](x) * self.scales[self.active]


def inject_lora(
    model: nn.Module, name: str, rank: int, alpha: float | None = None, targets=DEFAULT_TARGETS
) -> list[nn.Parameter]:
    """Add adapter ``name`` to every Linear under ``targets``; returns its parameters."""
    alpha = float(rank) if alpha is None else alpha
    params: list[nn.Parameter] = []

    def attach(layer: LoRALinear) -> None:
        if name not in layer.adapters:
            layer.add_adapter(name, rank, alpha)
            params.extend(layer.adapters[name].parameters())

    replaced: dict[int, nn.Module] = {}

    def walk(module: nn.Module) -> None:
        for child_name, child in module.named_children():
            if isinstance(child, LoRALinear):
                attach(child)  # already wrapped: never descend into base/adapters
            elif isinstance(child, nn.Linear):
                wrapped = LoRALinear(child)
                replaced[id(child)] = wrapped
                setattr(module, child_name, wrapped)
                attach(wrapped)
            else:
                walk(child)

    for target in targets:
        root = model.get_submodule(target)
        if isinstance(root, nn.Linear | LoRALinear):
            raise ValueError(f"target '{target}' is itself a Linear; name its parent instead")
        walk(root)
    # shared adaLN cores are also held, unregistered, by every block
    rebind_shared_cores(model, replaced)
    return params


def lora_layers(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def rebind_shared_cores(model: nn.Module, replaced: dict[int, nn.Module]) -> None:
    """Point every block's hidden core reference at its replacement.

    Shared cores live in ``model.modulation_cores`` and, unregistered, in each
    block's ``_core`` tuple. Code that swaps a registered core module (a LoRA
    wrapper) must call this, or the blocks keep calling the old one.
    ``replaced`` maps ``id(old_module)`` to the new module.
    """
    for module in model.modules():
        core = getattr(module, "_core", None)
        if isinstance(core, tuple) and core and id(core[0]) in replaced:
            module._core = (replaced[id(core[0])],)


@contextmanager
def use_adapter(model: nn.Module, name: str | None):
    """Activate adapter ``name``; ``None`` or ``"off"`` selects the bare base."""
    if name == "off":
        name = None
    layers = lora_layers(model).values()
    previous = [layer.active for layer in layers]
    for layer in layers:
        layer.active = name
    try:
        yield
    finally:
        for layer, active in zip(layers, previous, strict=True):
            layer.active = active


def save_adapter(model: nn.Module, name: str, path: str | Path, metadata: dict | None = None) -> None:
    from safetensors.torch import save_file

    tensors, rank, alpha = {}, None, None
    for prefix, layer in lora_layers(model).items():
        down, up = layer.adapters[name]
        tensors[f"{prefix}.lora_down"] = down.weight.detach().float().cpu().contiguous()
        tensors[f"{prefix}.lora_up"] = up.weight.detach().float().cpu().contiguous()
        rank, alpha = down.out_features, layer.scales[name] * down.out_features
    meta = {"rank": str(rank), "alpha": str(alpha), "extra": json.dumps(metadata or {})}
    save_file(tensors, str(path), metadata=meta)


def load_adapter(model: nn.Module, name: str, path: str | Path) -> None:
    """Load a saved adapter into slot ``name`` (injected beforehand with the same rank)."""
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    layers = lora_layers(model)
    for prefix, layer in layers.items():
        down, up = layer.adapters[name]
        down.weight.data.copy_(tensors[f"{prefix}.lora_down"])
        up.weight.data.copy_(tensors[f"{prefix}.lora_up"])
    expected = {f"{p}.lora_{s}" for p in layers for s in ("down", "up")}
    if missing := set(tensors) - expected:
        raise KeyError(f"adapter file has weights for layers the model lacks: {sorted(missing)[:5]}")


def read_metadata(path: str | Path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    return {"rank": int(meta["rank"]), "alpha": float(meta["alpha"]), **json.loads(meta.get("extra", "{}"))}


@torch.no_grad()
def merge_lora_file(model: nn.Module, path: str | Path, strength: float = 1.0) -> dict:
    """Fold an adapter file into a plain (uninjected) model's Linear weights."""
    from safetensors.torch import load_file

    meta = read_metadata(path)
    scale = strength * meta["alpha"] / meta["rank"]
    tensors = load_file(str(path))
    prefixes = {key.rsplit(".", 1)[0] for key in tensors}
    for prefix in prefixes:
        linear = model.get_submodule(prefix)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{prefix} is {type(linear).__name__}, expected Linear")
        down = tensors[f"{prefix}.lora_down"].to(linear.weight.device, torch.float32)
        up = tensors[f"{prefix}.lora_up"].to(linear.weight.device, torch.float32)
        merged = linear.weight.float() + scale * (up @ down)
        linear.weight.copy_(merged.to(linear.weight.dtype))
    return meta