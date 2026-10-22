"""Load DataSpaceIris from an exported Iris-3B directory without fp32 ever touching the GPU."""

import torch

from .dit import DataSpaceIris


def load_dataspace_iris(checkpoint: str, device="cuda", weights=None, raw=None):
    """Return ``(model, cfg)``: matrices bf16, 1-D params (norms, biases) fp32, new branches fp32 + zero.

    Pass ``weights``/``raw`` (from ``load_for_inference``) to reuse an already-read checkpoint.
    """
    from iris3b.config import inference_config
    from iris3b.sampling import load_for_inference

    if weights is None:
        raw, weights = load_for_inference(checkpoint)
    cfg = inference_config(raw, [])
    with torch.device("meta"):
        model = DataSpaceIris(cfg.model)
    # the base checkpoint knows nothing about the new branches, so exactly those keys are missing
    missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    if set(missing) != set(DataSpaceIris.BRANCH_KEYS) or unexpected:
        raise RuntimeError(f"checkpoint/model key mismatch: missing={missing} unexpected={unexpected}")
    for branch in (model.x0_embedder, model.step_embeddings, model.fake_step_embeddings):
        branch.to_empty(device=device)
    model.reset_dataspace_init()
    with torch.no_grad():
        for name, p in model.named_parameters():
            keep_fp32 = p.ndim < 2 or name in DataSpaceIris.BRANCH_KEYS
            p.data = p.data.to(device=device, dtype=torch.float32 if keep_fp32 else torch.bfloat16)
        for buf in model.buffers():
            buf.data = buf.data.to(device)
    stray = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if stray:
        raise RuntimeError(f"tensors still on meta: {stray[:5]}")
    return model.eval(), cfg
