"""Segmentation head on PixelSeal's own detector (ConvNeXt-tiny), message-conditioned.

Input 512 image -> 256 (antialiased 2x down, = what the decoder does) -> ConvNeXt stages (64,32,16,8 px grids) ->
FPN head -> 64x64 logits -> 128x128 grid. Message conditioning: per-position, per-bit agreement of the detector's
per-position logits with the known message (256 ch at 8x8), its mean (1 ch) and the global agreement (broadcast)."""
import sys; sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research")
import os; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from common import *
import torch.nn as nn

_K = torch.tensor([1., 3., 3., 1.]) / 8


def down2(x):
    """Exact equivalent of F.interpolate(..., scale 1/2, bilinear, antialias=True) on 512 -> 256 (triangle filter)."""
    k = _K.to(x.device, x.dtype)
    c = x.shape[1]
    x = F.pad(x, (1, 1, 1, 1), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, 4).expand(c, 1, 1, 4), stride=(1, 2), groups=c)
    x = F.conv2d(x, k.view(1, 1, 4, 1).expand(c, 1, 4, 1), stride=(2, 1), groups=c)
    return x


class DetFeatures(nn.Module):
    def __init__(self, det, hires=False):
        super().__init__(); self.det = det; self.hires = hires

    def _one(self, x, sgn):
        x = self.det.preprocess(x)
        cn = self.det.convnext; feats = []
        for i in range(4):
            x = cn.downsample_layers[i](x); x = cn.stages[i](x); feats.append(x)
        emb = self.det.pixel_decoder.output_upscaling(x)
        lin = self.det.pixel_decoder.linear
        pp = torch.einsum("bchw,oc->bohw", emb, lin.weight[1:]) + lin.bias[1:][None, :, None, None]   # B x 256 x h x w
        pa = pp * sgn[:, :, None, None]                                                              # per-bit agreement
        return feats, pa, (pp.mean((2, 3)) * sgn).mean(1)

    def forward(self, x512, sgn):
        """x512 B x 3 x 512 x 512 in [0,1]; sgn B x 256 in {-1,+1}. Returns a list of (feats, pa, glob) per stream;
        stream 0 = 256 input (the decoder's own scale), stream 1 (hires) = native 512 input."""
        out = [self._one(down2(x512), sgn)]
        if self.hires:
            out.append(self._one(x512, sgn))
        return out


class Head(nn.Module):
    def __init__(self, dims=(96, 192, 384, 768), c=64, streams=1):
        super().__init__()
        self.streams = streams
        self.lat = nn.ModuleList([nn.Conv2d(d, c, 1) for _ in range(streams) for d in dims])
        self.msg = nn.ModuleList([nn.Sequential(nn.Conv2d(256 + 2, c, 1), nn.GELU(), nn.Conv2d(c, c, 1)) for _ in range(streams)])
        self.fuse = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.GELU(),
                                  nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, 1, 1))

    def forward(self, streams):
        tgt = streams[-1][0][0].shape[-2:]                  # finest stage-1 grid (64 for 256 input, 128 for 512)
        glob = streams[0][2]
        y = 0
        for si, (feats, pa, _) in enumerate(streams):
            for f, l in zip(feats, self.lat[4 * si: 4 * si + 4]):
                y = y + F.interpolate(l(f), size=tgt, mode="bilinear", align_corners=False)
            g = glob[:, None, None, None].expand(-1, 1, *pa.shape[-2:])
            m = self.msg[si](torch.cat([pa / 10, pa.mean(1, keepdim=True), g], 1))
            y = y + F.interpolate(m, size=tgt, mode="bilinear", align_corners=False)
        lg = self.fuse(y)
        return F.interpolate(lg, size=(128, 128), mode="bilinear", align_corners=False)[:, 0]


def load_shards(dirs, max_shards=0):
    """Local shards (msgs/full) or the Modal eaic-gen shards (messages as strings, regimes as strings)."""
    X, Y, M, R = [], [], [], []
    paths = [p for d in str(dirs).split(",") for p in sorted(Path(d).glob("shard_*.npz"))]
    if max_shards: paths = paths[:max_shards]
    for p in paths:
        z = np.load(p); X.append(z["images"]); Y.append(z["masks128"])
        if "msgs" in z.files:
            M.append(z["msgs"]); R.append(z["full"])
        else:
            M.append(np.array([[int(c) for c in s] for s in z["messages"]], np.uint8)); R.append(z["regimes"] == "full")
    return np.concatenate(X), np.concatenate(Y), np.concatenate(M), np.concatenate(R)


def loss_fn(lg, yb):
    p = torch.sigmoid(lg)
    bce = F.binary_cross_entropy_with_logits(lg, yb)
    i1 = (p * yb).sum((1, 2)); u1 = (p + yb - p * yb).sum((1, 2))
    i0 = ((1 - p) * (1 - yb)).sum((1, 2)); u0 = ((1 - p) + (1 - yb) - (1 - p) * (1 - yb)).sum((1, 2))
    return bce + (1 - 0.5 * (i1 / (u1 + 1e-6) + i0 / (u0 + 1e-6))).mean()


def best_thr(probs, gts):
    best = (0.0, 0.5)
    for thr in np.arange(0.05, 0.96, 0.05):
        best = max(best, (score([fix_grid(p, thr) for p in probs], gts), float(thr)))
    return best


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--unfreeze", type=int, default=0, help="step at which the backbone is unfrozen (0 = never)")
    ap.add_argument("--blr", type=float, default=5e-5); ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data")); ap.add_argument("--tag", default="head")
    ap.add_argument("--hires", type=int, default=0, help="1: add a second detector pass on the native 512 image")
    ap.add_argument("--max_shards", type=int, default=0); ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "runs"))
    ap.add_argument("--test", type=int, default=0, help="write test-set probabilities with the best checkpoint")
    a = ap.parse_args()
    torch.manual_seed(0)
    wm = load_wm(); dev = DEV
    # sanity: down2 == antialiased bilinear
    xt = torch.rand(1, 3, 512, 512)
    ref = F.interpolate(xt, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
    print("down2 max err", float((down2(xt) - ref).abs().max()))
    X, Y, M, R = load_shards(a.data, a.max_shards)
    print(f"train: {len(X)} samples, full {R.mean():.2f}, marked {Y.mean():.3f}")
    vmeta, vimgs, vgts, vmsgs = official("train", 1)
    vfull = np.array([e["full_watermark"] for e in vmeta])
    base = score([np.ones_like(g) if f else np.zeros_like(g) for g, f in zip(vgts, vfull)], vgts)
    feat = DetFeatures(wm.det, hires=bool(a.hires)).to(dev); head = Head(streams=1 + a.hires).to(dev)
    for p in feat.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)

    def run(xb_u8, msg_u8, train):
        x = torch.from_numpy(xb_u8).to(dev).permute(0, 3, 1, 2).float().div(255)
        sgn = torch.from_numpy(msg_u8).to(dev).float() * 2 - 1
        if train and feat_trainable[0]:
            st = feat(x, sgn)
        else:
            with torch.no_grad():
                st = feat(x, sgn)
        return head(st), st[0][2]

    @torch.no_grad()
    def predict(imgs, msgs):
        head.eval(); feat.eval(); out, gl = [], []
        for i in range(0, len(imgs), 16):
            lg, g = run(imgs[i:i + 16], msgs[i:i + 16].astype(np.uint8), False)
            out.append(torch.sigmoid(lg).cpu().numpy()); gl.append(g.cpu().numpy())
        head.train(); return np.concatenate(out), np.concatenate(gl)

    feat_trainable = [False]; t0 = time.time(); best = (-1, 0.5, 0)
    outd = Path(a.out) / a.tag; outd.mkdir(parents=True, exist_ok=True)
    for step in range(1, a.steps + 1):
        if a.unfreeze and step == a.unfreeze:
            for p in feat.parameters(): p.requires_grad_(True)
            opt.add_param_group({"params": list(feat.parameters()), "lr": a.blr}); feat_trainable[0] = True
            print("backbone unfrozen")
        idx = rng.integers(0, len(X), a.bs)
        xb = X[idx]
        if rng.random() < 0.5: xb = xb[:, :, ::-1]; yb = Y[idx][:, :, ::-1]           # h-flip (residual is a fn of image, ok)
        else: yb = Y[idx]
        lg, _ = run(np.ascontiguousarray(xb), M[idx], True)
        loss = loss_fn(lg, torch.from_numpy(np.ascontiguousarray(yb)).to(dev).float())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"step {step} loss {float(loss):.3f} ({(time.time() - t0) / step:.2f}s/step)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            pr, gl = predict(vimgs, vmsgs)
            sc, thr = best_thr(pr, vgts)
            scf, _ = best_thr(pr[vfull], vgts[vfull]); scp, _ = best_thr(pr[~vfull], vgts[~vfull])
            print(f"== step {step}: val-half mIoU {sc:.4f} (thr {thr:.2f}) | full {scf:.3f} partial {scp:.3f} | regime baseline {base:.4f}", flush=True)
            if sc > best[0]:
                best = (sc, thr, step)
                torch.save({"head": head.state_dict(), "det": feat.det.state_dict() if feat_trainable[0] else None, "thr": thr}, outd / "best.pt")
                np.savez_compressed(outd / "val_probs.npz", names=np.array([e["image_name"] for e in vmeta]), probs=pr.astype(np.float16), glob=gl)
    print("best", best, flush=True)
    if a.test:
        ck = torch.load(outd / "best.pt", map_location=dev); head.load_state_dict(ck["head"])
        if ck["det"] is not None: feat.det.load_state_dict(ck["det"])
        tmeta, timgs, _, tmsgs = official("test", None)
        pr, gl = predict(timgs, tmsgs)
        np.savez_compressed(outd / "test_probs.npz", names=np.array([e["image_name"] for e in tmeta]), probs=pr.astype(np.float16),
                            glob=gl, thr=ck["thr"])
        print("test probs written", outd / "test_probs.npz", flush=True)
