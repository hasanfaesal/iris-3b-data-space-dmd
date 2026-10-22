"""Stage 1 check: the tiny fixture builds from real config classes and runs."""

import torch

from tiny import TINY_PATCH, tiny_iris, tiny_cond, tiny_model_config


def test_forward_shapes_square_and_buckets():
    model = tiny_iris().eval()
    # p=4: grids 4x4, 3x5, 2x7 -- square plus two aspect-bucket-like shapes
    for height, width in [(16, 16), (12, 20), (8, 28)]:
        x = torch.randn(2, 3, height, width)
        y, mask = tiny_cond()
        t = torch.full((2,), 1000.0)
        with torch.no_grad():
            out = model(x, t, y, y_mask=mask)
        assert out.x.shape == (2, 3, height, width)
        assert torch.isfinite(out.x).all()


def test_fixture_covers_paths_our_code_touches():
    cfg = tiny_model_config()
    model = tiny_iris()
    # shared modulation cores exist and single-stream blocks alias onto the img core
    assert set(model.modulation_cores.keys()) == {"adaln_img", "adaln_txt"}
    # hybrid trunk: dual-stream first, single-stream tail
    from iris3b.models.blocks.mmdit import MMDiTBlock
    from iris3b.models.blocks.single_stream import SingleStreamBlock
    from iris3b.models.blocks.pit import PiTBlock

    assert isinstance(model.blocks[0], MMDiTBlock)
    assert isinstance(model.blocks[cfg.dual_depth], SingleStreamBlock)
    assert all(isinstance(b, PiTBlock) for b in model.pixel_blocks)
    # the Linear inventory LoRA will wrap is non-trivial and includes the cores
    from iris3b.nn.modulation import SharedCoreBias

    n_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    assert n_linear > 20
    for core in model.modulation_cores.values():
        assert isinstance(core, torch.nn.Linear)
    # blocks reference the shared cores through the unregistered _core tuple
    mod = model.blocks[cfg.dual_depth].adaln
    assert isinstance(mod, SharedCoreBias) and mod.core is model.modulation_cores["adaln_img"]