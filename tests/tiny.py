"""Tiny CPU IrisDiT fixture built from the real package config classes.

Exercises the architectural paths our code touches: hybrid trunk (dual + single
streams), shared_bias modulation cores, GQA, gated attention, masked text
adapter, PiT head. Random weights, deterministic per seed, forward in <1 s.
"""

import torch
from dataclasses import replace

from iris3b.config import ModelConfig, PixelStageConfig
from iris3b.models.dit import IrisDiT
from src.models.dit import DataSpaceIris

TINY_TEXT_DIM = 32
TINY_TEXT_LEN = 16
TINY_PATCH = 4
BRANCH_PREFIXES = ("x0_embedder", "step_embeddings", "fake_step_embeddings")


def tiny_model_config(**overrides) -> ModelConfig:
    cfg = ModelConfig(
        block="single_stream",
        dual_depth=2,  # 2 dual-stream + 2 single-stream, like the real 8 + 16 hybrid
        depth=4,
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,  # GQA split q/k/v projections, like the real model
        gated_attention=True,  # wraps attn_gate Linears too
        patch_size=TINY_PATCH,
        modulation="shared_bias",  # shared cores adaln_img / adaln_txt, like the real model
        adaln_zero_init=False,  # fixture mimics a TRAINED model: live gates, real text mixing
        rope_aspect="isotropic",  # correct on non-square grids (aspect buckets)
        text_dim=TINY_TEXT_DIM,
        text_len=TINY_TEXT_LEN,
        text_adapter="blocks2",  # masked transformer adapter; cheaper stand-in for lap_blocks2
        pixel=PixelStageConfig(depth=1, hidden_size=8, attn_hidden_size=64, num_heads=2, modulation="post"),
    )
    return replace(cfg, **overrides)


def tiny_cond(batch: int = 2, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """(y, y_mask) for the tiny model: [B, 16, 32] states + [B, 16] mask."""
    y = torch.randn(batch, TINY_TEXT_LEN, TINY_TEXT_DIM, device=device, dtype=dtype)
    mask = torch.ones(batch, TINY_TEXT_LEN, dtype=torch.int64, device=device)
    return y, mask


def tiny_iris(seed: int = 0, **overrides) -> IrisDiT:
    """Random-weight tiny model on CPU."""
    torch.manual_seed(seed)
    with torch.device("meta"):
        model = IrisDiT(tiny_model_config(**overrides))
    model.to_empty(device="cpu")
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.02)
    return model


def dataspace_iris(seed: int = 0, **overrides) -> DataSpaceIris:
    """Tiny DataSpaceIris: random base weights, ZERO branch modules.

    Weight scale 0.2 (not the untrained 0.02): the adaLN gates and t_embedder
    of an untrained model are tiny, which suppresses attention mixing and with
    it the text->image path that the CFG tests must see.
    """
    torch.manual_seed(seed)
    model = DataSpaceIris(tiny_model_config(**overrides))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not name.startswith(BRANCH_PREFIXES):
                p.normal_(0, 0.2)
    return model.eval()