"""Do official irregular zones follow SAM masks of their own image? Median best IoU per zone component,
against the image's own SAM masks and (control) against SAM masks of another random image."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

R = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release/train")
SAM = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/amil/Downloads/hackathon-watermark/modal_t1/tracks/sam_shapes/sam")


def load_sam(split):
    out = {}
    for p in sorted(SAM.glob(f"{split}_*.npz")):
        z = np.load(p)
        for i, n in enumerate(z["names"].tolist()):
            sel = z["img"] == i
            out[n] = (z["bits"][sel], z["area"][sel], z["piou"][sel], z["stab"][sel])
    return out


def unpack(bits):
    return np.unpackbits(bits, axis=-1)[..., :512 * 512].reshape(-1, 512, 512).astype(bool)


if __name__ == "__main__":
    sam = load_sam("train")
    meta = [e for e in json.loads((R / "metadata.json").read_text()) if e["image_name"] in sam]
    print(len(meta), "train images with SAM masks; masks/img median", np.median([len(sam[e["image_name"]][1]) for e in meta]))
    rng = np.random.default_rng(0)
    rows = []
    for e in meta:
        n = e["image_name"]
        mk = np.asarray(Image.open(R / "masks" / f"{n}.png").convert("L")) > 127
        z = ~mk if e["full_watermark"] else mk
        M = unpack(sam[n][0])
        other = meta[int(rng.integers(len(meta)))]["image_name"]
        Mo = unpack(sam[other][0])
        lab, k = ndimage.label(z)
        for c in range(1, k + 1):
            cm = lab == c
            if cm.mean() < 0.001:
                continue
            ys, xs = np.where(cm); fill = cm.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1))
            def best(MM):
                if len(MM) == 0:
                    return 0.0
                inter = (MM & cm).sum((1, 2)); uni = (MM | cm).sum((1, 2))
                return float((inter / uni).max())
            rows.append(dict(name=n, method=e["method"], full=e["full_watermark"], area=float(cm.mean()), fill=float(fill),
                             own=best(M), ctrl=best(Mo)))
    Path(SAM.parent / "overlap_rows.json").write_text(json.dumps(rows))
    import collections
    for key in ["all", "full", "partial"]:
        rr = [r for r in rows if r["fill"] < 0.97 and (key == "all" or (r["full"] == (key == "full")))]
        own = np.array([r["own"] for r in rr]); ctrl = np.array([r["ctrl"] for r in rr])
        print(f"{key:8s} irregular comps {len(rr):4d}: median best IoU own {np.median(own):.3f} ctrl {np.median(ctrl):.3f}"
              f" | own>=0.8: {(own >= 0.8).mean():.2f}  own>=0.9: {(own >= 0.9).mean():.2f}  ctrl>=0.8: {(ctrl >= 0.8).mean():.2f}")
    for m in ("pixelseal", "trustmark", "stegastamp"):
        rr = [r for r in rows if r["fill"] < 0.97 and r["method"] == m]
        own = np.array([r["own"] for r in rr])
        print(f"{m:10s} n {len(rr)} median own {np.median(own):.3f} own>=0.8 {(own >= 0.8).mean():.2f}")
    rr = [r for r in rows if r["fill"] >= 0.97]
    print("rect-like comps", len(rr), "median own", np.median([r["own"] for r in rr]).round(3))
