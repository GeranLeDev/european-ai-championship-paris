"""Train a small U-Net that predicts the 128x128 watermark grid from the 512x512 image.
One shared model for the 3 methods; extra constant input channels: method one-hot + global message agreement
(mean tanh(logit) * (2*bit-1) from the official decoder with the known message; strong 'mostly watermarked' cue).
Data: synthetic pairs (solution/synth.py) + train-fold real images (oversampled). Validation: held-out real fold,
official per-image mIoU, per method.
Usage: python solution/train_seg.py --device mps --epochs 10 [--synth solution/synth] [--out solution/ckpt/seg.pt]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, METHODS, split_train
from wmloc.utils import score


def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, extra=4, w=(32, 64, 96, 128, 192, 256)):
        super().__init__()
        self.stem = nn.Sequential(nn.PixelUnshuffle(4), cbr(6 * 16, w[1] * 2), cbr(w[1] * 2, w[1]))  # 512 -> 128 (space-to-depth)
        self.enc = nn.ModuleList([nn.Sequential(cbr(w[i] + (extra if i == 1 else 0), w[i + 1], 2 if i > 1 else 1), cbr(w[i + 1], w[i + 1]))
                                  for i in range(1, 5)])                                                    # 128,64,32,16
        self.glob = nn.Sequential(nn.Linear(w[5] + extra, w[5]), nn.ReLU(inplace=True))
        self.dec = nn.ModuleList([nn.Sequential(cbr(w[i + 1] + w[i], w[i]), cbr(w[i], w[i])) for i in range(4, 1, -1)])
        self.head = nn.Conv2d(w[2], 1, 1)

    def forward(self, x, extra):
        hp = x - F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, 1)       # high-pass residual
        h = self.stem(torch.cat([x - 0.5, hp * 10], 1))
        e = extra[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])
        feats = []; h = torch.cat([h, e], 1)
        for i, blk in enumerate(self.enc):
            h = blk(h); feats.append(h)
        g = self.glob(torch.cat([h.mean((2, 3)), extra], 1))
        h = h + g[:, :, None, None]
        for blk, skip in zip(self.dec, feats[-2::-1]):
            h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
            h = blk(torch.cat([h, skip], 1))
        return self.head(h)[:, 0]                                                  # B x 128 x 128 logits


def extra_vec(method, agree):
    v = np.zeros(4, np.float32); v[METHODS.index(method)] = 1; v[3] = agree; return v


def miou_loss(logit, y):
    p = torch.sigmoid(logit); l = 0
    for pc, yc in ((p, y), (1 - p, 1 - y)):
        inter = (pc * yc).sum((1, 2)); union = (pc + yc - pc * yc).sum((1, 2))
        l = l + 1 - (inter + 1) / (union + 1)
    return (l / 2).mean()


def load_real():
    meta = json.loads((ROOT / "data" / "train" / "metadata.json").read_text())
    f = np.load(ROOT / "solution" / "cache" / "feat_train.npz"); agree = dict(zip(f["names"].tolist(), f["agree"]))
    z = np.load(ROOT / "data" / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    out = []
    for e in meta:
        n = e["image_name"]
        img = np.asarray(Image.open(ROOT / "data" / "train" / "images" / f"{n}.png").convert("RGB"))
        out.append(dict(name=n, method=e["method"], img=img, mask=gt[n], extra=extra_vec(e["method"], agree[n])))
    return out


@torch.no_grad()
def predict_probs(model, imgs, extras, device, bs=8, tta=True):
    model.eval(); out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack(imgs[i:i + bs])).permute(0, 3, 1, 2).float().div(255).to(device)
        e = torch.from_numpy(np.stack(extras[i:i + bs])).to(device)
        p = torch.sigmoid(model(x, e))
        if tta:
            p = (p + torch.sigmoid(model(x.flip(3), e)).flip(2)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def evaluate(model, items, device, thr=0.5):
    probs = predict_probs(model, [it["img"] for it in items], [it["extra"] for it in items], device)
    res = {}
    for m in METHODS:
        idx = [i for i, it in enumerate(items) if it["method"] == m]
        res[m] = score([(probs[i] > thr).astype(np.uint8) for i in idx], [items[i]["mask"] for i in idx])
    res["all"] = float(np.mean([res[m] for m in METHODS]))
    return res, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps"); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--synth", default=str(ROOT / "solution" / "synth")); ap.add_argument("--real_rep", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "solution" / "ckpt" / "seg.pt")); ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max_steps_per_epoch", type=int, default=0); ap.add_argument("--init", default="")
    a = ap.parse_args()
    torch.manual_seed(0); rng = np.random.default_rng(0)
    val_names = split_train()
    real = load_real()
    tr_real = [it for it in real if it["name"] not in val_names]; va_real = [it for it in real if it["name"] in val_names]
    synth = []
    for sd, m in [(sd, m) for sd in a.synth.split(",") for m in METHODS]:   # --synth dir1,dir2,...
        p = Path(sd) / f"{m}_meta.npz"
        if p.exists():
            md = np.load(p); arr = np.load(Path(sd) / f"{m}_images.npy", mmap_mode="r")
            synth += [(m, arr, j, md["masks"][j], extra_vec(m, md["agree"][j])) for j in range(len(md["masks"]))]
    print(f"real train {len(tr_real)}, val {len(va_real)}, synth {len(synth)}", flush=True)
    samples = [("real", it) for it in tr_real] * a.real_rep + [("syn", s) for s in synth]

    def get(s):
        if s[0] == "real":
            it = s[1]; return it["img"], it["mask"], it["extra"]
        m, arr, j, mask, ex = s[1]; return np.asarray(arr[j]), mask, ex

    dev = torch.device(a.device)
    model = UNet().to(dev)
    if a.init: model.load_state_dict(torch.load(a.init, map_location=dev))
    steps = len(samples) // a.bs
    if a.max_steps_per_epoch: steps = min(steps, a.max_steps_per_epoch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps * a.epochs, pct_start=0.1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    best = -1; t0 = time.time()
    for ep in range(a.epochs):
        model.train(); perm = rng.permutation(len(samples)); tl = 0
        for st in range(steps):
            batch = [get(samples[k]) for k in perm[st * a.bs:(st + 1) * a.bs]]
            x = torch.from_numpy(np.stack([b[0] for b in batch])).permute(0, 3, 1, 2).float().div(255)
            y = torch.from_numpy(np.stack([b[1] for b in batch])).float()
            e = torch.from_numpy(np.stack([b[2] for b in batch]))
            if rng.random() < 0.5: x, y = x.flip(3), y.flip(2)
            x, y, e = x.to(dev), y.to(dev), e.to(dev)
            logit = model(x, e)
            loss = F.binary_cross_entropy_with_logits(logit, y) + miou_loss(logit, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
            if st % 100 == 0:
                print(f"ep {ep} step {st}/{steps} loss {tl / (st + 1):.4f} {time.time() - t0:.0f}s", flush=True)
        res, _ = evaluate(model, va_real, dev)
        print(f"== ep {ep} val {json.dumps({k: round(v, 4) for k, v in res.items()})} {time.time() - t0:.0f}s", flush=True)
        torch.save(model.state_dict(), a.out.replace(".pt", "_last.pt"))
        if res["all"] > best:
            best = res["all"]; torch.save(model.state_dict(), a.out)
    print("best val", best)


if __name__ == "__main__":
    main()
