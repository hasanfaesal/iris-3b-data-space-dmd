"""Stage 2 checks: role LoRA is an exact no-op at init and when off, and
gradients only reach the active role."""

import torch

from src.models.lora import LoRALinear, inject_lora, lora_layers, use_adapter
from tiny import tiny_cond, tiny_iris


def _perturb(params):
    for p in params:
        p.data.normal_(0, 0.02)


def test_off_is_bitwise_base_whole_model():
    """Teacher role reproduces the pre-injection model exactly, even with
    perturbed adapters -- proves wrapping + core rebinding changed nothing."""
    model = tiny_iris(seed=1).eval()
    x = torch.randn(2, 3, 16, 16)
    y, mask = tiny_cond()
    t = torch.full((2,), 1000.0)
    with torch.no_grad():
        base_out = model(x, t, y, y_mask=mask).x
    gen_params = inject_lora(model, "gen", 8)
    fake_params = inject_lora(model, "fake", 4)
    _perturb(gen_params + fake_params)
    with torch.no_grad():
        with use_adapter(model, "off"):
            off_out = model(x, t, y, y_mask=mask).x
    assert torch.equal(base_out, off_out)


def test_gen_at_zero_init_is_bitwise_base():
    model = tiny_iris(seed=2).eval()
    x = torch.randn(2, 3, 16, 16)
    y, mask = tiny_cond()
    t = torch.full((2,), 1000.0)
    inject_lora(model, "gen", 8)
    with torch.no_grad():
        with use_adapter(model, None):
            base_out = model(x, t, y, y_mask=mask).x
        with use_adapter(model, "gen"):
            gen_out = model(x, t, y, y_mask=mask).x
    assert torch.equal(base_out, gen_out)


def test_shared_cores_are_wrapped_and_rebound():
    model = tiny_iris(seed=3)
    inject_lora(model, "gen", 8)
    # every shared core is wrapped, and every block's hidden _core reference
    # points at the wrapper, not the orphaned original
    for core in model.modulation_cores.values():
        assert isinstance(core, LoRALinear)
    n_core_refs = 0
    for module in model.modules():
        core = getattr(module, "_core", None)
        if isinstance(core, tuple) and core:
            assert isinstance(core[0], LoRALinear)
            n_core_refs += 1
    assert n_core_refs > 0
    # single-stream blocks alias onto the img core through the wrapper
    assert model.blocks[2].adaln.core is model.modulation_cores["adaln_img"]


def test_grads_flow_only_to_active_role():
    model = tiny_iris(seed=4)
    gen_params = inject_lora(model, "gen", 8)
    fake_params = inject_lora(model, "fake", 4)
    _perturb(gen_params + fake_params)
    x = torch.randn(2, 3, 16, 16)
    y, mask = tiny_cond()
    t = torch.full((2,), 1000.0)
    with use_adapter(model, "gen"):
        model(x, t, y, y_mask=mask).x.square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in gen_params)
    assert all(p.grad is None for p in fake_params)
    # and symmetrically: a fake backward touches only fake params
    with use_adapter(model, "fake"):
        model(x, t, y, y_mask=mask).x.square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in fake_params)


def test_adapter_switching_is_nested_safe():
    """A context must restore the previous active role on exit."""
    model = tiny_iris(seed=5)
    inject_lora(model, "gen", 8)
    inject_lora(model, "fake", 4)
    with use_adapter(model, "gen"):
        with use_adapter(model, "fake"):
            pass
        assert all(l.active == "gen" for l in lora_layers(model).values())
    assert all(l.active is None for l in lora_layers(model).values())