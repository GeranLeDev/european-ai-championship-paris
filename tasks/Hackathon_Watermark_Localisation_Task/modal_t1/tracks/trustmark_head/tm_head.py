"""Segmentation head on TrustMark's own decoder (ResNet-50, input 245), message-conditioned.

512 image -> 245 (antialiased bilinear, x*2-1 = the decoder's own preprocessing) -> dilated (a trous, output stride 8)
copy of the frozen decoder: layer1 62x62, layer2/3/4 31x31 + per-position logits (fc on layer4) 100 x 31 x 31.
Message conditioning: per-position, per-bit agreement with the known message, its mean, the global agreement.
FPN head at 62 -> 128, refined at 128 with a small RGB + high-pass stem of the 512 image (object-shaped boundaries).
The decoder only decodes at its native scale (checked: 512/768/1024 inputs give 50% bits), so one stream only."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/root/ps"); sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research")
import os; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from common import *  # noqa
import torch.nn as nn
from tm_common import dilate_resnet, resnet_feats

PA_SCALE = float(os.environ.get("TM_PA_SCALE", "8.0"))


class TMFeatures(nn.Module):
    def __init__(self, net, os_=8):
        super().__init__(); self.net = dilate_resnet(net, os_)

    def forward(self, x512, sgn):
        x = F.interpolate(x512, size=(245, 245), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
        feats = resnet_feats(self.net, x)
        fc = self.net.fc
        pp = torch.einsum("bchw,oc->bohw", feats[-1], fc.weight) + fc.bias[None, :, None, None]   # B x 100 x 31 x 31
        pa = pp * sgn[:, :, None, None]
        gl = pp.mean((2, 3))
        glob = (gl * sgn).mean(1)                         # soft global agreement
        acc = ((gl > 0).float() * 2 - 1 == sgn).float().mean(1)   # global bit accuracy (dense GAP)
        return feats, pa, glob, acc


def gblur_k(dev):
    ax = torch.arange(5, device=dev).float() - 2
    g1 = torch.exp(-ax ** 2 / 2); g1 = g1 / g1.sum()
    return (g1[:, None] * g1[None, :]).expand(3, 1, 5, 5).contiguous()


class Head(nn.Module):
    def __init__(self, dims=(256, 512, 1024, 2048), c=96, nbits=100, stem=1):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(d, c, 1) for d in dims])
        self.msg = nn.Sequential(nn.Conv2d(nbits + 3, c, 1), nn.GELU(), nn.Conv2d(c, c, 1))
        self.fuse = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.GELU(),
                                  nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.GELU())
        self.use_stem = bool(stem)
        cs = 32 if stem else 0
        if stem:
            self.stem = nn.Sequential(nn.Conv2d(6, 24, 3, stride=2, padding=1), nn.GELU(),
                                      nn.Conv2d(24, cs, 3, stride=2, padding=1), nn.GroupNorm(8, cs), nn.GELU())
        self.ref = nn.Sequential(nn.Conv2d(c + cs, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
                                 nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, 1, 1))
        self.register_buffer("blur", torch.zeros(3, 1, 5, 5), persistent=False)

    def forward(self, x512, st):
        feats, pa, glob, acc = st
        tgt = feats[0].shape[-2:]
        y = 0
        for f, l in zip(feats, self.lat):
            y = y + F.interpolate(l(f), size=tgt, mode="bilinear", align_corners=False)
        g = glob[:, None, None, None].expand(-1, 1, *pa.shape[-2:]); a = acc[:, None, None, None].expand(-1, 1, *pa.shape[-2:])
        m = self.msg(torch.cat([torch.tanh(pa / PA_SCALE), pa.mean(1, keepdim=True) / PA_SCALE, g / PA_SCALE, a * 2 - 1], 1))
        y = self.fuse(y + F.interpolate(m, size=tgt, mode="bilinear", align_corners=False))
        y = F.interpolate(y, size=(128, 128), mode="bilinear", align_corners=False)
        if self.use_stem:
            if self.blur.abs().sum() == 0:
                self.blur = gblur_k(x512.device)
            hp = (x512 - F.conv2d(F.pad(x512, (2, 2, 2, 2), mode="reflect"), self.blur, groups=3)) * 20
            y = torch.cat([y, self.stem(torch.cat([(x512 - 0.45) / 0.25, hp], 1))], 1)
        return self.ref(y)[:, 0]


def load_shards(dirs, max_shards=0):
    """eaic-gen shards (messages as strings, regimes as strings). max_shards applies to EACH dir."""
    X, Y, M, R = [], [], [], []
    paths = []
    for d in str(dirs).split(","):
        ps = sorted(Path(d).glob("shard_*.npz"))
        paths += ps[:max_shards] if max_shards else ps
    t0 = time.time()
    for p in paths:
        z = np.load(p); X.append(z["images"]); Y.append(z["masks128"])
        M.append(np.array([[int(c) for c in s] for s in z["messages"]], np.uint8)); R.append(z["regimes"] == "full")
    print(f"loaded {len(paths)} shards in {time.time() - t0:.0f}s", flush=True)
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
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--unfreeze", type=int, default=0); ap.add_argument("--blr", type=float, default=3e-5)
    ap.add_argument("--eval_every", type=int, default=500); ap.add_argument("--data", required=True); ap.add_argument("--tag", default="tmhead")
    ap.add_argument("--max_shards", type=int, default=0); ap.add_argument("--out", default=str(HERE / "runs"))
    ap.add_argument("--stem", type=int, default=1); ap.add_argument("--os", type=int, default=8)
    ap.add_argument("--test", type=int, default=0); ap.add_argument("--val_limit", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(0); dev = DEV
    wm = load_watermark("trustmark", MODELS, dev)
    X, Y, M, R = load_shards(a.data, a.max_shards)
    print(f"train: {len(X)} samples, full {R.mean():.2f}, marked {Y.mean():.3f}", flush=True)
    vmeta, vimgs, vgts, vmsgs = official("train", 1, "trustmark")
    if a.val_limit: vmeta, vimgs, vgts, vmsgs = vmeta[:a.val_limit], vimgs[:a.val_limit], vgts[:a.val_limit], vmsgs[:a.val_limit]
    vfull = np.array([e["full_watermark"] for e in vmeta])
    base = score([np.ones_like(g) if f else np.zeros_like(g) for g, f in zip(vgts, vfull)], vgts)
    feat = TMFeatures(wm.net, a.os).to(dev).eval(); head = Head(stem=a.stem).to(dev)
    for p in feat.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    import math
    warm = max(1, int(0.05 * a.steps))
    rng = np.random.default_rng(0)
    trainable = [False]

    def run(xb_u8, msg_u8, train):
        x = torch.from_numpy(np.ascontiguousarray(xb_u8)).to(dev).permute(0, 3, 1, 2).float().div(255)
        sgn = torch.from_numpy(np.ascontiguousarray(msg_u8)).to(dev).float() * 2 - 1
        if train and trainable[0]:
            st = feat(x, sgn)
        else:
            with torch.no_grad():
                st = feat(x, sgn)
        return head(x, st), st

    @torch.no_grad()
    def predict(imgs, msgs):
        head.eval(); out, gl = [], []
        for i in range(0, len(imgs), 16):
            lg, st = run(imgs[i:i + 16], msgs[i:i + 16].astype(np.uint8), False)
            out.append(torch.sigmoid(lg).float().cpu().numpy()); gl.append(st[3].cpu().numpy())
        head.train(); return np.concatenate(out), np.concatenate(gl)

    t0 = time.time(); best = (-1, 0.5, 0)
    outd = Path(a.out) / a.tag; outd.mkdir(parents=True, exist_ok=True)
    for step in range(1, a.steps + 1):
        if a.unfreeze and step == a.unfreeze:
            for p in feat.net.parameters(): p.requires_grad_(True)
            opt.add_param_group({"params": list(feat.net.parameters()), "lr": a.blr})
            trainable[0] = True; print("backbone unfrozen (BN kept in eval mode)", flush=True)
        opt.param_groups[0]["lr"] = a.lr * min(1.0, step / warm) * 0.5 * (1 + math.cos(math.pi * step / a.steps)) + 1e-6
        idx = rng.integers(0, len(X), a.bs)
        lg, _ = run(X[idx], M[idx], True)
        loss = loss_fn(lg, torch.from_numpy(Y[idx]).to(dev).float())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"step {step} loss {float(loss):.3f} ({(time.time() - t0) / step:.3f}s/step)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            pr, gl = predict(vimgs, vmsgs)
            sc, thr = best_thr(pr, vgts)
            scf, _ = best_thr(pr[vfull], vgts[vfull]); scp, _ = best_thr(pr[~vfull], vgts[~vfull])
            print(f"== step {step}: val-half mIoU {sc:.4f} (thr {thr:.2f}) | full {scf:.3f} partial {scp:.3f} | regime baseline {base:.4f}", flush=True)
            if sc > best[0]:
                best = (sc, thr, step)
                torch.save({"head": head.state_dict(), "net": feat.net.state_dict() if trainable[0] else None, "thr": thr, "os": a.os, "stem": a.stem}, outd / "best.pt")
                np.savez_compressed(outd / "val_probs.npz", names=np.array([e["image_name"] for e in vmeta]), probs=pr.astype(np.float16), glob=gl)
    print("best", best, flush=True)
    if a.test:
        ck = torch.load(outd / "best.pt", map_location=dev); head.load_state_dict(ck["head"])
        if ck["net"] is not None: feat.net.load_state_dict(ck["net"])
        tmeta, timgs, _, tmsgs = official("test", None, "trustmark")
        pr, gl = predict(timgs, tmsgs)
        np.savez_compressed(outd / "test_probs.npz", names=np.array([e["image_name"] for e in tmeta]), probs=pr.astype(np.float16),
                            glob=gl, thr=ck["thr"])
        print("test probs written", outd / "test_probs.npz", len(tmeta), flush=True)
