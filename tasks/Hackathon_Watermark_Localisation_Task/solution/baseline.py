"""Non-learned baseline (no training data synthesis):
  1. global decision per image: mean signed agreement between the decoder's bit logits and the known message
     -> 'mostly watermarked' (full mode) vs 'mostly clean'
  2. within each mode, threshold the local agreement map from per_position_logits (pixelseal, trustmark; 8x8 upsampled)
Thresholds are tuned per method on the train fold, evaluated with the official mIoU on the held-out fold.
Needs solution/cache/feat_{train,test}.npz (solution/features.py)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, METHODS, split_train
from wmloc.utils import score


def load(split):
    f = np.load(ROOT / "solution" / "cache" / f"feat_{split}.npz")
    meta = {e["image_name"]: e for e in json.loads((ROOT / "data" / split / "metadata.json").read_text())}
    return f, meta


def predict(agree, ppl, tg, tf, tp):
    if agree > tg:
        return (ppl > tf).astype(np.uint8) if tf is not None else np.ones((128, 128), np.uint8)
    return (ppl > tp).astype(np.uint8) if tp is not None else np.zeros((128, 128), np.uint8)


def main():
    f, meta = load("train")
    z = np.load(ROOT / "data" / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    val = split_train()
    names = f["names"].tolist(); ppl = f["ppl"].astype(np.float32)
    params, report = {}, {}
    for m in METHODS:
        tr = [i for i, n in enumerate(names) if meta[n]["method"] == m and n not in val]
        va = [i for i, n in enumerate(names) if meta[n]["method"] == m and n in val]
        ev = lambda idx, p: score([predict(f["agree"][i], ppl[i], *p) for i in idx], [gt[names[i]] for i in idx])
        tgs = np.linspace(-0.2, 0.8, 51)
        tg = max(tgs, key=lambda t: ev(tr, (t, None, None)))
        best = (tg, None, None); bs = ev(tr, best)
        if m != "stegastamp":
            for tf in [None] + list(np.linspace(-0.5, 0.8, 27)):
                for tp in [None] + list(np.linspace(-0.2, 0.9, 23)):
                    s = ev(tr, (tg, tf, tp))
                    if s > bs: bs, best = s, (tg, tf, tp)
        params[m] = best
        report[m] = dict(train=bs, val_mode_only=ev(va, (tg, None, None)), val=ev(va, best),
                         val_zeros=ev(va, (9, None, None)), val_oracle_mode=score(
                             [np.full((128, 128), int(meta[names[i]]["full_watermark"]), np.uint8) for i in va], [gt[names[i]] for i in va]))
        print(m, best, {k: round(v, 4) for k, v in report[m].items()})
    print("overall val:", np.mean([report[m]["val"] for m in METHODS]))
    json.dump({m: [None if v is None else float(v) for v in p] for m, p in params.items()},
              open(ROOT / "solution" / "cache" / "baseline_params.json", "w"))


if __name__ == "__main__":
    main()
