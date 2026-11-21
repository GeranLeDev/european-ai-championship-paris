"""Sliding-window decoding: decode every window (size s, stride st) with the official decoder, signed agreement with
the known message, accumulate per 128-grid cell (mean over covering windows). Returns 128x128 maps per window size."""
import sys; from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wmloc.utils import to_tensor


def window_map(wm, img, msg, size=128, stride=32, bs=64):
    x = to_tensor(img)
    s = torch.from_numpy(2 * msg.astype(np.float32) - 1)
    pos = [(y, xx) for y in range(0, 512 - size + 1, stride) for xx in range(0, 512 - size + 1, stride)]
    crops = torch.cat([x[:, :, y:y + size, xx:xx + size] for y, xx in pos])
    ag = []
    for i in range(0, len(crops), bs):
        lg = wm.logits(crops[i:i + bs]).cpu()
        ag.append((torch.tanh(lg) * s).mean(1))
    ag = torch.cat(ag).numpy()
    acc = np.zeros((128, 128), np.float32); cnt = np.zeros((128, 128), np.float32)
    for a, (y, xx) in zip(ag, pos):
        acc[y // 4:(y + size) // 4, xx // 4:(xx + size) // 4] += a; cnt[y // 4:(y + size) // 4, xx // 4:(xx + size) // 4] += 1
    return acc / np.maximum(cnt, 1)


if __name__ == "__main__":
    # python solution/sliding.py --device cuda --methods trustmark,pixelseal,stegastamp --splits train,test
    import argparse, json
    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark
    from sklearn.metrics import roc_auc_score
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); ap.add_argument("--methods", default="pixelseal,trustmark,stegastamp")
    ap.add_argument("--splits", default="train"); ap.add_argument("--sizes", default="64,128,256"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); sizes = [int(v) for v in a.sizes.split(",")]
    out = ROOT / "solution" / "cache"; out.mkdir(parents=True, exist_ok=True)
    for split in a.splits.split(","):
        ds = LocalisationDataset(ROOT / "data", split)
        for method in a.methods.split(","):
            wm = load_watermark(method, ROOT / "models", a.device)
            idx = [i for i, m in enumerate(ds.meta) if m["method"] == method]
            if a.limit: idx = idx[:a.limit]
            names, maps, auc = [], [], {}
            for i in idx:
                it = ds[i]
                mp = np.stack([window_map(wm, it["image"], it["message"], sz, sz // 4) for sz in sizes])
                names.append(it["image_name"]); maps.append(mp.astype(np.float16))
                if split == "train":
                    g = it["mask128"].reshape(-1)
                    if g.min() != g.max():
                        for k, sz in enumerate(sizes):
                            auc.setdefault((it["full_watermark"], sz), []).append(roc_auc_score(g, mp[k].reshape(-1)))
            np.savez_compressed(out / f"slide_{split}_{method}.npz", names=np.array(names), maps=np.stack(maps), sizes=np.array(sizes))
            for k, v in sorted(auc.items()):
                print(split, method, "full" if k[0] else "partial", "win", k[1], "AUC", round(float(np.mean(v)), 3), flush=True)
            print("done", split, method, len(idx), flush=True)
