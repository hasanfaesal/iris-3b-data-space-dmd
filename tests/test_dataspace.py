"""Stage 3 checks: data-space branches are exact no-ops at init and off, and
gradients follow the corrected rules (B and step tables on step 1; everything
after B becomes nonzero)."""

import torch

from src.models.dit import DataSpaceIris
from src.models.lora import inject_lora, lora_layers, use_adapter
from tiny import BRANCH_PREFIXES, dataspace_iris, tiny_cond


def _perturb_branches(model, scale=0.05):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith(BRANCH_PREFIXES):
                p.normal_(0, scale)


def _perturb_adapters(params):
    with torch.no_grad():
        for p in params:
            p.data.normal_(0, 0.02)


def _base_out(model, x, t, y, mask):
    with torch.no_grad():
        with use_adapter(model, None):
            return model(x, t, y, y_mask=mask).x


def _inputs():
    x = torch.randn(2, 3, 16, 16)
    y, mask = tiny_cond()
    t = torch.full((2,), 1000.0)
    return x, t, y, mask


def test_gen_zero_init_is_bitwise_base_for_any_x0():
    """Zero-init branch contributes exactly 0, so gen == base even with random x0."""
    model = dataspace_iris(seed=1)
    inject_lora(model, "gen", 8)
    x, t, y, mask = _inputs()
    x0 = torch.randn_like(x)
    with torch.no_grad():
        base = _base_out(model, x, t, y, mask)
        with use_adapter(model, "gen"):
            gen = model(x, t, y, x0=x0, r=1, y_mask=mask).x
    assert torch.equal(base, gen)


def test_teacher_ignores_branch_weights():
    """'off' output must not change when branch weights go from zero to random."""
    model = dataspace_iris(seed=2)
    inject_lora(model, "gen", 8)
    inject_lora(model, "fake", 4)
    x, t, y, mask = _inputs()
    with torch.no_grad():
        with use_adapter(model, "off"):
            out_zero_branches = model(x, t, y, y_mask=mask).x
    _perturb_branches(model, scale=10.0)  # huge, so any leak is obvious
    with torch.no_grad():
        with use_adapter(model, "off"):
            out_perturbed = model(x, t, y, y_mask=mask).x
    assert torch.equal(out_zero_branches, out_perturbed)


def test_teacher_rejects_x0_and_r():
    model = dataspace_iris(seed=3)
    x, t, y, mask = _inputs()
    with use_adapter(model, "off"):
        try:
            model(x, t, y, x0=torch.zeros_like(x), r=0, y_mask=mask)
        except ValueError:
            pass
        else:
            raise AssertionError("teacher forward must reject x0/r")


def test_gen_branches_change_output():
    """With perturbed branches: x0 matters, and r=0 vs r=1 differ."""
    model = dataspace_iris(seed=4)
    inject_lora(model, "gen", 8)
    _perturb_adapters([p for n, p in model.named_parameters() if n.startswith(("x0_embedder", "step_embeddings"))])
    x, t, y, mask = _inputs()
    with torch.no_grad():
        with use_adapter(model, "gen"):
            out_x0 = model(x, t, y, x0=torch.randn_like(x), r=1, y_mask=mask).x
            out_zero = model(x, t, y, x0=torch.zeros_like(x), r=1, y_mask=mask).x
            out_r0 = model(x, t, y, x0=torch.randn_like(x), r=0, y_mask=mask).x
    assert not torch.equal(out_x0, out_zero)
    assert not torch.equal(out_x0, out_r0)


def test_fake_uses_only_its_own_table():
    """Fresh fake adapter == base bitwise even with poisoned x0 branch and gen
    table; perturbing only the fake table makes r matter."""
    model = dataspace_iris(seed=5)
    inject_lora(model, "fake", 4)
    x, t, y, mask = _inputs()
    with torch.no_grad():  # poison x0_embedder + gen table; fake table stays zero
        model.x0_embedder.weight.normal_(0, 10.0)
        model.step_embeddings.weight.normal_(0, 10.0)
    with torch.no_grad():
        with use_adapter(model, "fake"):
            out = model(x, t, y, r=2, y_mask=mask).x
    assert torch.equal(_base_out(model, x, t, y, mask), out)
    with torch.no_grad():  # now give the fake table its own signal
        model.fake_step_embeddings.weight.normal_(0, 0.05)
        with use_adapter(model, "fake"):
            out_r1 = model(x, t, y, r=1, y_mask=mask).x
            out_r2 = model(x, t, y, r=2, y_mask=mask).x
    assert not torch.equal(out_r1, out_r2)


def test_grad_rules_step1_b_zero_x0_zero():
    """The plan's corrected rule, part 1: at step 1 (B=0) with x0=0, r=0 --
    A matrices and x0_embedder get exactly zero grad; B and the used step row
    get nonzero grad."""
    model = dataspace_iris(seed=6)
    gen_params = inject_lora(model, "gen", 8)
    a_params = [p for p in gen_params if p.shape[0] == next(q.shape[0] for q in gen_params)]
    x = torch.randn(2, 3, 16, 16)
    y, mask = tiny_cond()
    t = torch.full((2,), 1000.0)
    with use_adapter(model, "gen"):
        model(x, t, y, x0=torch.zeros_like(x), r=0, y_mask=mask).x.square().mean().backward()
    assert model.x0_embedder.weight.grad is not None
    assert model.x0_embedder.weight.grad.abs().sum() == 0  # exact: input was 0
    step_grad = model.step_embeddings.weight.grad
    assert step_grad is not None and step_grad[0].abs().sum() > 0
    assert step_grad[1:].abs().sum() == 0  # only r=0 was used
    down_params = [m.adapters["gen"][0].weight for m in lora_layers(model).values()]
    up_params = [m.adapters["gen"][1].weight for m in lora_layers(model).values()]
    assert all(p.grad is not None and p.grad.abs().sum() == 0 for p in down_params)  # B=0 kills dA
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in up_params)


def test_grad_rules_after_b_nonzero():
    """The plan's corrected rule, part 2: once B is nonzero (i.e. after
    optimizer steps) and x0 is nonzero at r>=1, everything trains: A, B,
    x0_embedder, and the used step row."""
    model = dataspace_iris(seed=7)
    inject_lora(model, "gen", 8)
    for m in lora_layers(model).values():  # simulate post-update B
        with torch.no_grad():
            m.adapters["gen"][1].weight.normal_(0, 0.02)
    x = torch.randn(2, 3, 16, 16)
    x0 = torch.randn_like(x)
    y, mask = tiny_cond()
    t = torch.full((2,), 1000.0)
    with use_adapter(model, "gen"):
        model(x, t, y, x0=x0, r=1, y_mask=mask).x.square().mean().backward()
    assert model.x0_embedder.weight.grad.abs().sum() > 0
    step_grad = model.step_embeddings.weight.grad
    assert step_grad[1].abs().sum() > 0 and step_grad[[0, 2, 3]].abs().sum() == 0
    down_params = [m.adapters["gen"][0].weight for m in lora_layers(model).values()]
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in down_params)


def test_fake_grads_hit_only_fake_table():
    model = dataspace_iris(seed=8)
    inject_lora(model, "fake", 4)
    x, t, y, mask = _inputs()
    with use_adapter(model, "fake"):
        model(x, t, y, r=3, y_mask=mask).x.square().mean().backward()
    fg = model.fake_step_embeddings.weight.grad
    assert fg is not None and fg[3].abs().sum() > 0 and fg[:3].abs().sum() == 0
    assert model.step_embeddings.weight.grad is None  # gen table not in graph
    assert model.x0_embedder.weight.grad is None  # x0 branch not in graph


def test_mixed_adapter_state_raises():
    model = dataspace_iris(seed=9)
    inject_lora(model, "gen", 8)
    inject_lora(model, "fake", 4)
    first = next(iter(lora_layers(model).values()))
    first.active = "gen"  # corrupt one layer manually
    x, t, y, mask = _inputs()
    try:
        model(x, t, y, r=0, y_mask=mask)
    except RuntimeError:
        pass
    else:
        raise AssertionError("mixed adapter state must raise")
    finally:
        first.active = None