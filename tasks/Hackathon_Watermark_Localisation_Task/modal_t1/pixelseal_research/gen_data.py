"""Generate PixelSeal training examples locally with the official recipe (gen_lib.make_sample) on val2017 photos.
Shards: images (N,512,512,3) uint8, masks128 (N,128,128) uint8, msgs (N,256) uint8, full (N,) bool."""
import sys; sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research")
import os; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from common import *
from gen_lib import load_clean, make_sample

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2400); ap.add_argument("--shard", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wm = load_wm(); rng = np.random.default_rng(a.seed)
    files = sorted(Path("/Users/amil/Downloads/val2017").glob("*.jpg")); rng.shuffle(files)
    buf = dict(images=[], masks128=[], msgs=[], full=[]); k = 0; t = time.time()
    for i in range(a.n):
        img, _ = load_clean(Image.open(files[i % len(files)]), rng)
        w, mask, msg, reg = make_sample(wm, "pixelseal", img, rng)
        g = downsample_mask(mask.astype(np.uint8))
        if g.sum() == 0 or g.sum() == g.size:
            continue
        buf["images"].append(w); buf["masks128"].append(g); buf["msgs"].append(msg.astype(np.uint8)); buf["full"].append(reg == "full")
        if len(buf["images"]) == a.shard:
            np.savez(out / f"shard_s{a.seed}_{k:03d}.npz", **{kk: np.stack(v) for kk, v in buf.items()})
            print(f"shard {k} ({i + 1} samples, {time.time() - t:.0f}s)", flush=True); k += 1
            buf = {kk: [] for kk in buf}
