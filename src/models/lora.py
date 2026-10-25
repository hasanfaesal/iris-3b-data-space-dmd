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

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
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
        tensors[f"{prefix}.lora_down.weight"] = down.weight.detach().float().cpu().contiguous()
        tensors[f"{prefix}.lora_up.weight"] = up.weight.detach().float().cpu().contiguous()
        rank, alpha = down.out_features, layer.scales[name] * down.out_features

    if hasattr(model, "x0_embedder") and hasattr(model.x0_embedder, "weight"):
        tensors["x0_embedder.weight"] = model.x0_embedder.weight.detach().float().cpu().contiguous()
    if hasattr(model, "step_embeddings") and hasattr(model.step_embeddings, "weight"):
        tensors["step_embeddings.weight"] = model.step_embeddings.weight.detach().float().cpu().contiguous()

    meta_in = metadata or {}
    num_steps = tensors["step_embeddings.weight"].shape[0] if "step_embeddings.weight" in tensors else 4
    meta = {
        "rank": str(rank if rank is not None else meta_in.get("rank", "64")),
        "alpha": str(alpha if alpha is not None else meta_in.get("alpha", "64.0")),
        "training_step": str(meta_in.get("training_step", 0)),
        "num_steps": str(meta_in.get("num_steps", num_steps)),
        "source_checkpoint": str(meta_in.get("source_checkpoint", "")),
        "date": str(meta_in.get("date", datetime.now(timezone.utc).isoformat())),
        "model_type": str(meta_in.get("model_type", "iris-3b-dataspace-turbo")),
        "extra": json.dumps(meta_in),
    }
    save_file(tensors, str(path), metadata=meta)


def load_adapter(model: nn.Module, name: str, path: str | Path) -> None:
    """Load a saved adapter into slot ``name`` (injected beforehand with the same rank)."""
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    layers = lora_layers(model)
    for prefix, layer in layers.items():
        down, up = layer.adapters[name]
        down_w = tensors.get(f"{prefix}.lora_down.weight", tensors.get(f"{prefix}.lora_down"))
        up_w = tensors.get(f"{prefix}.lora_up.weight", tensors.get(f"{prefix}.lora_up"))
        if down_w is not None:
            down.weight.data.copy_(down_w)
        if up_w is not None:
            up.weight.data.copy_(up_w)

    if "x0_embedder.weight" in tensors and hasattr(model, "x0_embedder") and hasattr(model.x0_embedder, "weight"):
        model.x0_embedder.weight.data.copy_(
            tensors["x0_embedder.weight"].to(device=model.x0_embedder.weight.device, dtype=model.x0_embedder.weight.dtype)
        )
    if "step_embeddings.weight" in tensors and hasattr(model, "step_embeddings") and hasattr(model.step_embeddings, "weight"):
        model.step_embeddings.weight.data.copy_(
            tensors["step_embeddings.weight"].to(device=model.step_embeddings.weight.device, dtype=model.step_embeddings.weight.dtype)
        )


def read_metadata(path: str | Path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    res = {}
    if "rank" in meta:
        res["rank"] = int(meta["rank"])
    if "alpha" in meta:
        res["alpha"] = float(meta["alpha"])
    if "training_step" in meta:
        res["training_step"] = int(meta["training_step"])
    if "num_steps" in meta:
        res["num_steps"] = int(meta["num_steps"])
    if "source_checkpoint" in meta:
        res["source_checkpoint"] = meta["source_checkpoint"]
    if "date" in meta:
        res["date"] = meta["date"]
    if "model_type" in meta:
        res["model_type"] = meta["model_type"]
    if "extra" in meta:
        res.update(json.loads(meta["extra"]))
    return res


def extract_adapter_from_checkpoint(
    checkpoint: dict | str | Path,
    weights_key: str = "ema",
    rank: int | None = None,
    alpha: float | None = None,
    source_name: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Extract adapter tensors and safetensors metadata dict from a checkpoint."""
    if isinstance(checkpoint, (str, Path)):
        source_name = source_name or Path(checkpoint).name
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    else:
        state = checkpoint
        source_name = source_name or "checkpoint.pt"

    weights = state.get(weights_key, state.get("ema", state.get("gen", state))) if isinstance(state, dict) else state
    if not isinstance(weights, dict):
        raise ValueError(f"Expected dict of weights, got {type(weights)} in checkpoint")

    tensors: dict[str, torch.Tensor] = {}
    detected_rank = rank
    for name, tensor in weights.items():
        if ".adapters.gen.0.weight" in name:
            prefix = name.replace(".adapters.gen.0.weight", "")
            tensors[f"{prefix}.lora_down.weight"] = tensor.detach().float().cpu().contiguous()
            if detected_rank is None:
                detected_rank = tensor.shape[0]
        elif ".adapters.gen.1.weight" in name:
            prefix = name.replace(".adapters.gen.1.weight", "")
            tensors[f"{prefix}.lora_up.weight"] = tensor.detach().float().cpu().contiguous()
        elif name == "x0_embedder.weight":
            tensors["x0_embedder.weight"] = tensor.detach().float().cpu().contiguous()
        elif name == "step_embeddings.weight":
            tensors["step_embeddings.weight"] = tensor.detach().float().cpu().contiguous()

    if not tensors:
        raise ValueError(f"No generator LoRA or branch weights found under key '{weights_key}'")

    rank_final = detected_rank if detected_rank is not None else 64
    alpha_final = alpha if alpha is not None else float(rank_final)
    training_step = state.get("gen_updates", 0) if isinstance(state, dict) else 0
    num_steps = tensors["step_embeddings.weight"].shape[0] if "step_embeddings.weight" in tensors else 4

    metadata = {
        "rank": str(rank_final),
        "alpha": str(alpha_final),
        "training_step": str(training_step),
        "num_steps": str(num_steps),
        "source_checkpoint": str(source_name),
        "date": datetime.now(timezone.utc).isoformat(),
        "model_type": "iris-3b-dataspace-turbo",
    }
    return tensors, metadata


def save_adapter_file(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
    out_path: str | Path,
) -> None:
    from safetensors.torch import save_file

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=metadata)


@torch.no_grad()
def merge_lora_file(model: nn.Module, path: str | Path, strength: float = 1.0) -> dict:
    """Fold an adapter file into a plain (uninjected) model's Linear weights."""
    from safetensors.torch import load_file

    meta = read_metadata(path)
    scale = strength * meta["alpha"] / meta["rank"]
    tensors = load_file(str(path))

    if "x0_embedder.weight" in tensors and hasattr(model, "x0_embedder") and hasattr(model.x0_embedder, "weight"):
        model.x0_embedder.weight.data.copy_(
            tensors["x0_embedder.weight"].to(device=model.x0_embedder.weight.device, dtype=model.x0_embedder.weight.dtype)
        )
    if "step_embeddings.weight" in tensors and hasattr(model, "step_embeddings") and hasattr(model.step_embeddings, "weight"):
        model.step_embeddings.weight.data.copy_(
            tensors["step_embeddings.weight"].to(device=model.step_embeddings.weight.device, dtype=model.step_embeddings.weight.dtype)
        )

    prefixes = {
        key.replace(".lora_down.weight", "").replace(".lora_up.weight", "").replace(".lora_down", "").replace(".lora_up", "")
        for key in tensors
        if "lora" in key
    }
    for prefix in prefixes:
        linear = model.get_submodule(prefix)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{prefix} is {type(linear).__name__}, expected Linear")
        down_w = tensors.get(f"{prefix}.lora_down.weight", tensors.get(f"{prefix}.lora_down"))
        up_w = tensors.get(f"{prefix}.lora_up.weight", tensors.get(f"{prefix}.lora_up"))
        if down_w is None or up_w is None:
            continue
        down = down_w.to(linear.weight.device, torch.float32)
        up = up_w.to(linear.weight.device, torch.float32)
        merged = linear.weight.float() + scale * (up @ down)
        linear.weight.copy_(merged.to(linear.weight.dtype))
    return meta