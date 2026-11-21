"""Sliding-window message agreement with PixelSeal's own detector.
The image is downsampled to the embedder's 256 working resolution (watermark pattern back at its trained scale), then
small windows are fed to the fully-convolutional detector on their own; per window: agreement of the logits with the
known message. Per-pixel map = mean of the scores of the windows covering it."""
import sys; sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research")
import os; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from common import *

OUT = Path(__file__).resolve().parent / "cache"; OUT.mkdir(exist_ok=True)


def resize(x, s):
    return F.interpolate(x.cpu(), size=(s, s), mode="bilinear", align_corners=False, antialias=True).to(x.device)


@torch.no_grad()
def window_map(wm, x, msg, w, stride, bs=512):
    """x: 1x3xSxS (already at the working scale). Returns SxS map of per-window agreement (mean over bits of
    logit * sign) averaged over covering windows, plus bit-acc map."""
    S = x.shape[-1]
    pat = F.unfold(x, w, stride=stride)                         # 1 x (3ww) x L
    L = pat.shape[-1]; n = (S - w) // stride + 1
    crops = pat[0].T.reshape(L, 3, w, w)
    sgn = torch.as_tensor(2 * msg - 1, dtype=torch.float32, device=x.device)
    sc, acc = [], []
    for i in range(0, L, bs):
        lg = wm.det(crops[i:i + bs])[:, 1:].float()
        if lg.dim() > 2: lg = lg.mean((2, 3))
        sc.append((lg * sgn).mean(1)); acc.append(((lg > 0).float() == (sgn > 0).float()).float().mean(1))
    out = []
    for v in (torch.cat(sc), torch.cat(acc)):
        cols = v[None, None, :].expand(1, w * w, L)
        num = F.fold(cols, (S, S), w, stride=stride)
        den = F.fold(torch.ones_like(cols), (S, S), w, stride=stride)
        out.append((num / den.clamp_min(1))[0, 0])
    return torch.stack(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--parity", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scale", type=int, default=256); ap.add_argument("--wins", default="32,64,96,128"); ap.add_argument("--stride", type=int, default=8)
    a = ap.parse_args()
    wm = load_wm()
    meta, imgs, gts, msgs = official("train", a.parity)
    if a.limit: meta, imgs, gts, msgs = meta[:a.limit], imgs[:a.limit], gts[:a.limit], msgs[:a.limit]
    full = np.array([e["full_watermark"] for e in meta])
    wins = [int(v) for v in a.wins.split(",")]
    res = {w: [] for w in wins}; t = time.time()
    for i in range(len(meta)):
        x = resize(to_t(imgs[i:i + 1]), a.scale)
        for w in wins:
            m = window_map(wm, x, msgs[i], w, a.stride)            # 2 x S x S
            m = F.interpolate(m[None], size=(128, 128), mode="area")[0] if a.scale != 128 else m
            res[w].append(m.cpu().numpy().astype(np.float32))
    print(f"{len(meta)} images in {time.time() - t:.0f}s")
    np.savez_compressed(OUT / f"slide_p{a.parity}_s{a.scale}.npz", names=np.array([e["image_name"] for e in meta]), full=full, gts=gts,
                        **{f"w{w}": np.stack(res[w]) for w in wins})
    for w in wins:
        M = np.stack(res[w])
        for j, nm in enumerate(("agree", "bitacc")):
            r = []
            for reg, sel in (("full", full), ("partial", ~full)):
                aucs = [auc(g, s) for s, g in zip(M[sel, j], gts[sel])]
                r.append(f"{reg} AUC {np.mean(aucs):.3f}")
            print(f"scale {a.scale} win {w} {nm}: " + " | ".join(r))
