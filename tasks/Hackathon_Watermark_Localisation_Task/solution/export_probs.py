"""Convert <run>/pred.npz (solution/train_seg2.py) to the team exchange format of solution/ensemble.py:
    <out>/<pipeline>_<method>_{train,test}.npz with names, probs float16 [N,128,128] and (train) used_in_training.
used_in_training = the image was NOT in the held-out set of that run (its real train fold was used as samples and covers).
Writes atomically and re-reads every file (inode / quota incidents on the scratch).
Usage: python solution/export_probs.py --pipeline seg2 --out output/probs trustmark=output/seg2/trustmark pixelseal=... stegastamp=..."""
import argparse, os
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--pipeline", default="seg2"); ap.add_argument("--out", required=True); ap.add_argument("runs", nargs="+")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


def save(path, **arrs):
    tmp = str(path) + f".tmp{os.getpid()}"
    np.savez_compressed(tmp, **arrs); os.replace(tmp + ".npz", path)
    z = np.load(path); assert len(z["names"]) == len(z["probs"]) and z["probs"].shape[1:] == (128, 128), path
    print("wrote", path, len(z["names"]), {k: z[k].shape for k in z.files if k != "names"})


for kv in a.runs:
    m, d = kv.split("="); p = np.load(Path(d) / "pred.npz")
    sel = p["method"] == m if "method" in p.files else np.ones(len(p["names"]), bool)
    save(out / f"{a.pipeline}_{m}_train.npz", names=p["names"][sel], probs=p["probs"][sel].astype(np.float16), used_in_training=~p["val"][sel])
    if "test_names" in p.files:
        sel = p["test_method"] == m if "test_method" in p.files else np.ones(len(p["test_names"]), bool)
        save(out / f"{a.pipeline}_{m}_test.npz", names=p["test_names"][sel], probs=p["test_probs"][sel].astype(np.float16))
