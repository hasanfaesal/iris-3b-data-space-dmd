"""A/B vs the NOTES s21 teacher render: identical sampler, cfg=1.0 (no guidance).
A washed-out result confirms the student is faithfully matching an unguided
teacher (Problem 1: Beta(2.5,1)+shift puts ~96% of rows outside the CFG window).
Also renders cfg=3.0 and cfg=4.0 for a complete, self-contained comparison.
"""
import sys
from pathlib import Path
import torch
from PIL import Image

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
from iris3b.flow.solver import FlowDPMSolver  # noqa: E402
from src.data.embed_cache import EmbedCache, load_null  # noqa: E402
from src.models.loading import load_dataspace_iris  # noqa: E402

CKPT, CACHE = "/home/dev/models/iris-3b", "/home/dev/cache"
PROMPTS = [2, 4]  # the 02.png / 04.png used everywhere so far
STEPS, ORDER, SHIFT, SEED = 20, 2, 3.0, 0  # exactly the s21 settings

print("Loading model...", flush=True)
model, _ = load_dataspace_iris(CKPT)  # eval, bf16 base, no adapters => plain teacher
model = model.cuda()

print("Loading embeddings...", flush=True)
y, mask = EmbedCache(CACHE).get(PROMPTS)
y, mask = y.cuda(), mask.cuda()

null, null_mask = load_null(CACHE)
null = null.cuda().expand(len(PROMPTS), -1, -1, -1)
null_mask = null_mask.cuda().expand(len(PROMPTS), -1)
cfg_mask = torch.cat([null_mask, mask])  # solver doubles the batch as cat([uncond, cond])


def model_fn(x, t, cond):
    m = mask if cond is y else cfg_mask
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(x, t, cond, y_mask=m).x.float()


g = torch.Generator(device="cuda").manual_seed(SEED)  # same z as the s21 cfg=4.0 renders
z = torch.randn(len(PROMPTS), 3, 1024, 1024, device="cuda", generator=g).float()

out = Path("runs/nocfg_check")
out.mkdir(parents=True, exist_ok=True)

for cfg in (1.0, 3.0, 4.0):
    print(f"\n--- Sampling cfg={cfg} ---", flush=True)
    solver = FlowDPMSolver(model_fn, num_timesteps=1000, cfg_scale=cfg, prediction="v")
    sample = solver.sample(z, y, null if cfg != 1.0 else None, steps=STEPS, order=ORDER, shift=SHIFT).clamp(-1, 1)
    img = (sample + 1) / 2
    for j, idx in enumerate(PROMPTS):
        x = img[j]
        Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")).save(
            out / f"teacher_cfg{cfg}_{idx:02d}.png"
        )
        chroma = (x.amax(0) - x.amin(0)).mean()
        print(f"cfg={cfg} prompt {idx:02d}: pixel_std={x.std():.3f} chroma={chroma:.3f}", flush=True)
print("\nDone!", flush=True)
