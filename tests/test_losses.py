"""Stage 4 checks: CFG window, loss behavior, and the analytic DMD gradient."""

import torch

from src.engine.losses import dmd_generator_loss, fake_critic_loss, teacher_velocity_windowed
from src.engine.rollout import rollout_dataspace
from src.models.lora import inject_lora, lora_layers, use_adapter
from tiny import dataspace_iris, tiny_cond


def _model_with_both(seed=0):
    model = dataspace_iris(seed=seed)
    inject_lora(model, "gen", 8)
    inject_lora(model, "fake", 4)
    with torch.no_grad():  # make both adapters live
        for n, p in model.named_parameters():
            if ".adapters." in n and "lora_up" in n:
                p.normal_(0, 0.02)
    return model


def _inputs(batch=2):
    z = torch.randn(batch, 3, 16, 16)
    y, mask = tiny_cond(batch)
    null, null_mask = tiny_cond(batch)
    return z, y, mask, null, null_mask


def test_cfg_window_applies_only_below_threshold():
    model = _model_with_both()
    x_tau = torch.randn(2, 3, 16, 16)
    sigma = torch.tensor([0.3, 0.9])
    y, mask = tiny_cond(2)
    null, null_mask = tiny_cond(2)
    v_pos = teacher_velocity_windowed(model, x_tau, sigma, y, mask, null, null_mask, cfg_scale=3.0, cfg_window=0.6)
    t = sigma * 1000.0
    with torch.no_grad():
        with use_adapter(model, "off"):
            v_cond = model(x_tau, t, y, y_mask=mask).x
            v_uncond = model(x_tau, t, null, y_mask=null_mask).x
    assert not torch.allclose(v_cond, v_uncond, atol=1e-6)  # uncond must differ for the test to bite
    v_cfg = v_uncond + 3.0 * (v_cond - v_uncond)
    assert torch.allclose(v_pos[0], v_cfg[0], atol=1e-5)  # sigma' 0.3 <= 0.6: CFG on
    assert torch.allclose(v_pos[1], v_cond[1], atol=1e-5)  # sigma' 0.9 > 0.6: CFG off


def test_default_cfg_covers_high_noise_rows():
    """Regression: the default must guide every row, including sigma' 0.9 (the bulk of Beta-then-shift)."""
    model = _model_with_both()
    x_tau = torch.randn(2, 3, 16, 16)
    sigma = torch.tensor([0.3, 0.9])
    y, mask = tiny_cond(2)
    null, null_mask = tiny_cond(2)
    v_pos = teacher_velocity_windowed(model, x_tau, sigma, y, mask, null, null_mask, cfg_scale=3.0)
    t = sigma * 1000.0
    with torch.no_grad():
        with use_adapter(model, "off"):
            v_cond = model(x_tau, t, y, y_mask=mask).x
            v_uncond = model(x_tau, t, null, y_mask=null_mask).x
    assert torch.allclose(v_pos, v_uncond + 3.0 * (v_cond - v_uncond), atol=1e-5)


def test_cfg_scale_invariance_above_window():
    """Above the window the loss must not depend on cfg_scale; below, it must."""
    model = _model_with_both()
    z, y, mask, null, null_mask = _inputs()
    x0 = rollout_dataspace(model, z, y, mask, r=1, track_grad=False)
    noise = torch.randn_like(z)
    high = torch.full((2,), 0.9)
    low = torch.full((2,), 0.3)
    kw = dict(y=y, y_mask=mask, null=null, null_mask=null_mask, r=1, cfg_window=0.6)
    with torch.no_grad():
        l_hi_a = dmd_generator_loss(model, x0, high, noise, cfg_scale=3.0, **kw)
        l_hi_b = dmd_generator_loss(model, x0, high, noise, cfg_scale=1.0, **kw)
        l_lo_a = dmd_generator_loss(model, x0, low, noise, cfg_scale=3.0, **kw)
        l_lo_b = dmd_generator_loss(model, x0, low, noise, cfg_scale=1.0, **kw)
    assert torch.allclose(l_hi_a, l_hi_b)
    assert not torch.allclose(l_lo_a, l_lo_b)


def test_dmd_gradient_is_the_teacher_direction():
    """Analytic identity: dL/dx0 = (v_pos - v_fake)/numel, hence a gradient DESCENT
    step moves x0 along +(x0_teacher - x0_fake)/sigma, i.e. toward the
    teacher's clean prediction."""
    model = _model_with_both()
    z, y, mask, null, null_mask = _inputs()
    x0 = rollout_dataspace(model, z, y, mask, r=1, track_grad=False).requires_grad_(True)
    sigma = torch.full((2,), 0.3)
    noise = torch.randn_like(z)
    loss = dmd_generator_loss(model, x0, sigma, noise, y, mask, null, null_mask, r=1)
    loss.backward()
    per = sigma.view(-1, 1, 1, 1)
    x_tau = (1 - per) * x0.detach() + per * noise
    v_pos = teacher_velocity_windowed(model, x_tau, sigma, y, mask, null, null_mask)
    with torch.no_grad():
        with use_adapter(model, "fake"):
            v_fake = model(x_tau, sigma * 1000.0, y, r=1, y_mask=mask).x
    n = x0.numel()  # mean reduction over every element
    assert torch.allclose(x0.grad * n, v_pos - v_fake, atol=1e-5)
    x0_pos = x_tau - per * v_pos
    x0_fake = x_tau - per * v_fake
    assert torch.allclose(-x0.grad * n, (x0_pos - x0_fake) / per, atol=1e-5)


def test_fake_loss_trains_fake_only():
    model = _model_with_both()
    gen_up = [m.adapters["gen"][1].weight for m in lora_layers(model).values()]
    fake_up = [m.adapters["fake"][1].weight for m in lora_layers(model).values()]
    z, y, mask, _, _ = _inputs()
    sigma = torch.tensor([0.4, 0.7])
    noise = torch.randn_like(z)
    loss = fake_critic_loss(model, z, y, mask, r=2, sigma=sigma, noise=noise)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in fake_up)
    assert all(p.grad is None for p in gen_up)
    fg = model.fake_step_embeddings.weight.grad
    assert fg is not None and fg[2].abs().sum() > 0  # the used step row trains
    assert model.step_embeddings.weight.grad is None
    assert model.x0_embedder.weight.grad is None  # no x0 branch on the fake path

def test_pixel_only_checkpointing_matches_full_gradients():
    """Trunk un-checkpointed + checkpointed PiT head must give the same grads as `full`."""
    grads = {}
    for policy in ("full", "pixel_only"):
        model = _model_with_both()
        model.train()
        model.set_activation_checkpointing(policy)
        z, y, mask, null, null_mask = _inputs()
        torch.manual_seed(1)
        with use_adapter(model, "fake"):  # backward inside the adapter context, as the trainer does
            loss = fake_critic_loss(model, z, y, mask, r=2, sigma=torch.tensor([0.4, 0.7]), noise=torch.randn_like(z))
            loss.backward()
        grads[policy] = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert grads["full"].keys() == grads["pixel_only"].keys() and grads["full"]
    for n, g in grads["full"].items():
        assert torch.allclose(g, grads["pixel_only"][n], atol=1e-6), n
