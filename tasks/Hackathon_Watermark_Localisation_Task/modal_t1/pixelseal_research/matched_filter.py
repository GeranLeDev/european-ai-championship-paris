"""Training-free matched filter: re-embed the (watermarked) image with the KNOWN message to predict the residual
the embedder would have added, then measure local correlation between the image's high-pass and that prediction.
Random messages give the null. Saves per-image maps for fast post-processing experiments."""
import sys; sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research")
import os; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from common import *

OUT = Path(__file__).resolve().parent / "cache"; OUT.mkdir(exist_ok=True)


def hp(x, s=1.0):
    """luminance high-pass: x - gaussian_blur(x)."""
    k = int(2 * round(3 * s) + 1); ax = torch.arange(k, device=x.device).float() - k // 2
    g = torch.exp(-ax ** 2 / (2 * s * s)); g = g / g.sum()
    y = x.mean(1, keepdim=True)
    b = F.conv2d(F.pad(y, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k))
    b = F.conv2d(b, g.view(1, 1, k, 1))
    return y - b


def box(x, k):
    return F.avg_pool2d(F.pad(x, (k // 2, k - 1 - k // 2, k // 2, k - 1 - k // 2), mode="reflect"), k, 1)


@torch.no_grad()
def maps_for(wm, W, msg, n_rand=4, rng=None, iters=1, sigmas=(1.0,), ks=(8, 16, 32)):
    """W: 1x3x512x512 float. Returns dict of 512x512 maps."""
    msgs = np.concatenate([msg[None], rng.integers(0, 2, (n_rand, 256))]).astype(np.float32)
    Wb = W.expand(len(msgs), -1, -1, -1).contiguous()
    R = embed_float(wm, Wb, msgs) - Wb
    for _ in range(iters - 1):                       # refine: predict the residual from the estimated clean image
        Ih = (Wb - R).clamp(0, 1); R = embed_float(wm, Ih, msgs) - Ih
    Ry = R.mean(1, keepdim=True)
    out = {}
    for s in sigmas:
        e = hp(W, s)                                   # 1x1xHxW
        Rh = hp(R, s)                                  # same filter on the prediction
        for k in ks:
            num = box(e * Rh, k); den = torch.sqrt(box(e * e, k) * box(Rh * Rh, k)) + 1e-12
            rho = (num / den)[:, 0]                    # (1+n_rand) x H x W
            out[f"s{s}_k{k}"] = torch.stack([rho[0], rho[1:].mean(0), rho[1:].std(0)]).cpu().numpy().astype(np.float16)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--parity", type=int, default=1); ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    wm = load_wm(); rng = np.random.default_rng(0)
    meta, imgs, gts, msgs = official("train", a.parity)
    if a.limit: meta, imgs, gts, msgs = meta[:a.limit], imgs[:a.limit], gts[:a.limit], msgs[:a.limit]
    t = time.time(); allmaps = []
    for i in range(len(meta)):
        W = to_t(imgs[i:i + 1])
        allmaps.append(maps_for(wm, W, msgs[i], rng=rng, iters=a.iters))
    print(f"{len(meta)} images in {time.time() - t:.0f}s")
    keys = list(allmaps[0])
    np.savez_compressed(OUT / f"mf_p{a.parity}_it{a.iters}.npz", names=np.array([e["image_name"] for e in meta]),
                        full=np.array([e["full_watermark"] for e in meta]), gts=gts,
                        **{k: np.stack([m[k] for m in allmaps]) for k in keys})
    from common import auc as roc_auc_score
    full = np.array([e["full_watermark"] for e in meta])
    for k in keys:
        M = np.stack([m[k] for m in allmaps]).astype(np.float32)        # N x 3 x 512 x 512
        cell = M.reshape(len(M), 3, 128, 4, 128, 4).mean((3, 5))
        for name, sc in (("rho_m", cell[:, 0]), ("rho_m-rand", cell[:, 0] - cell[:, 1])):
            res = []
            for reg, sel in (("full", full), ("partial", ~full)):
                aucs = [roc_auc_score(g.ravel(), s.ravel()) for s, g in zip(sc[sel], gts[sel]) if 0 < g.mean() < 1]
                res.append(f"{reg} AUC/img {np.mean(aucs):.3f}")
            print(k, name, " | ".join(res))
