"""Evaluate a checkpoint on synthetic samples (official mIoU), split by method and mode: python solution/eval_synth.py ckpt synth_dir [n]"""
import sys; from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import METHODS
from train_seg import UNet, extra_vec, predict_probs
from wmloc.utils import score
ck, sd = sys.argv[1], Path(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
dev = sys.argv[4] if len(sys.argv) > 4 else "cpu"
model = UNet().to(dev); model.load_state_dict(torch.load(ck, map_location=dev))
for m in METHODS:
    md = np.load(sd / f"{m}_meta.npz"); arr = np.load(sd / f"{m}_images.npy", mmap_mode="r"); k = min(n, len(md["masks"]))
    p = predict_probs(model, [np.asarray(arr[j]) for j in range(k)], [extra_vec(m, md["agree"][j]) for j in range(k)], dev, tta=False)
    for mode in (0, 1):
        idx = [j for j in range(k) if md["modes"][j] == mode]
        const = np.full((128, 128), mode, np.uint8)
        print(m, "full" if mode else "partial", f"model {score([(p[j] > .5) for j in idx], [md['masks'][j] for j in idx]):.3f}",
              f"const {score([const] * len(idx), [md['masks'][j] for j in idx]):.3f}", flush=True)
