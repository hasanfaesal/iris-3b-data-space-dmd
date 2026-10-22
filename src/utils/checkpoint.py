"""Atomic torch checkpoint IO (write to a temp file, then rename)."""

import os
from pathlib import Path

import torch


def save_checkpoint(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # a crash mid-save never corrupts the previous checkpoint


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)  # holds RNG state objects
