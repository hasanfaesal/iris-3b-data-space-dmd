"""Atomic torch checkpoint IO (write to a temp file, then rename)."""

import errno
import os
import shutil
from pathlib import Path

import torch

# The rename keeps the previous file until the new one lands, so the .tmp is a
# second full copy. Leave 2 GiB: a full 250 GiB node volume makes
# ~/.givemeanode/commands unwritable and the next launch fails with exit 127.
KEEP_FREE = 2 << 30


def tensor_bytes(state) -> int:
    total = 0

    def walk(x) -> None:
        nonlocal total
        if torch.is_tensor(x):
            total += x.nbytes
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(state)
    return total


def save_checkpoint(state: dict, path: str | Path) -> bool:
    """Write ``state`` atomically. Returns False, and leaves any previous file, when the disk cannot take it."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        need = tensor_bytes(state) + KEEP_FREE
        free = shutil.disk_usage(path.parent).free
        if free < need:
            print(f"skip checkpoint {path}: need {need / 2**30:.1f} GiB, free {free / 2**30:.1f} GiB", flush=True)
            return False
        torch.save(state, tmp)
        os.replace(tmp, path)  # a crash mid-save never corrupts the previous checkpoint
    except OSError as e:
        tmp.unlink(missing_ok=True)
        if e.errno != errno.ENOSPC:
            raise
        print(f"skip checkpoint {path}: disk full", flush=True)
        return False
    return True


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)  # holds RNG state objects
