"""Shared helpers for task 3 (unlearning). Imports model builder / preprocessing from the official template."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import task_template as tt  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)


def pick_device(name=None):
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_device(dev):
    tt.DEVICE = dev  # template's preprocess() uses tt.DEVICE


def load_orig(dev):
    m = tt.build_model()
    m.load_state_dict(torch.load(ROOT / "model.pt", map_location="cpu", weights_only=True), strict=True)
    return m.to(dev).eval()


def load_forget():
    return tt.load_forget_images(ROOT / "forget_images.npz")


def to_input(u8, dev):
    """uint8 [B,H,W,3] -> normalized float [B,3,H,W] on dev (same as server)."""
    x = u8.to(dev).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


@torch.no_grad()
def logits_of(model, x_u8, dev, bs=64, aug=None):
    model.eval()
    out = []
    for i in range(0, len(x_u8), bs):
        x = to_input(x_u8[i:i + bs], dev)
        if aug is not None:
            x = aug(x)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out.append(model(x).float().cpu())
    return torch.cat(out)
