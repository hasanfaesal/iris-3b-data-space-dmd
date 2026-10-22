"""Stage 4 checks: rollout holds z fixed, grads only on the last step, r=0 is
a single forward, and the sigma sampler respects its clamp."""

import torch

from src.engine.rollout import rollout_dataspace, sample_shifted_sigma
from src.models.lora import inject_lora, lora_layers, use_adapter
from tiny import dataspace_iris, tiny_cond


class Recorder(torch.nn.Module):
    """Delegates to the model while recording every call's arguments."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls: list[dict] = []

    def forward(self, x, t, y, **kw):
        self.calls.append({"x": x, "x0": kw.get("x0"), "r": kw.get("r")})
        out = self.inner(x, t, y, **kw)
        self.calls[-1]["had_graph"] = out.x.grad_fn is not None
        return out


def _gen_model(seed=0):
    model = dataspace_iris(seed=seed)
    gen_params = inject_lora(model, "gen", 8)
    with torch.no_grad():  # make the adapter live so outputs differ per step
        for n, p in model.named_parameters():
            if ".adapters." in n and "lora_up" in n:
                p.normal_(0, 0.02)
    return model, gen_params


def test_rollout_holds_z_fixed_and_counts_calls():
    model, _ = _gen_model()
    rec = Recorder(model)
    z = torch.randn(2, 3, 16, 16)
    y, mask = tiny_cond()
    x_hat = rollout_dataspace(rec, z, y, mask, r=2)
    assert len(rec.calls) == 3
    for i, call in enumerate(rec.calls):
        assert torch.equal(call["x"], z)  # same z at every step
        assert call["r"] == i
        assert call["had_graph"] == (i == 2)  # grad only on the last call
        assert call["x0"].requires_grad is False  # states from no-grad steps
    # state evolution matches the manual recurrence
    x = torch.zeros_like(z)
    with torch.no_grad():
        with use_adapter(model, "gen"):
            for i in range(3):
                v = model(z, torch.full((2,), 1000.0), y, x0=x, r=i, y_mask=mask).x
                x = z - v
    assert torch.equal(x_hat, x)
    assert x_hat.requires_grad


def test_rollout_r0_is_single_forward():
    model, _ = _gen_model()
    z = torch.randn(2, 3, 16, 16)
    y, mask = tiny_cond()
    t_full = torch.full((2,), 1000.0)
    with torch.no_grad():
        with use_adapter(model, "gen"):
            expected = z - model(z, t_full, y, x0=torch.zeros_like(z), r=0, y_mask=mask).x
        got = rollout_dataspace(model, z, y, mask, r=0, track_grad=False)
    assert torch.equal(got, expected)


def test_rollout_grads_reach_gen_adapter():
    model, gen_params = _gen_model()
    z = torch.randn(2, 3, 16, 16)
    y, mask = tiny_cond()
    x_hat = rollout_dataspace(model, z, y, mask, r=1)
    x_hat.square().mean().backward()
    up_params = [m.adapters["gen"][1].weight for m in lora_layers(model).values()]
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in up_params)


def test_sigma_sampler_bounds_and_shape():
    sig = sample_shifted_sigma(256, shift=4.0, device="cpu")
    assert sig.shape == (256,)
    assert (sig >= 0.02).all() and (sig <= 0.98).all()
    # Beta(2.5, 1) shifted concentrates on high noise
    assert sig.mean() > 0.5