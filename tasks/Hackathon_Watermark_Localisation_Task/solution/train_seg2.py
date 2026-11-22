"""Full-resolution watermark segmenter, one model per watermark method.
Input 512 x 512 (never downscaled before the network): image, high-pass residual, global decoder agreement with the
known message, optionally the re-embedding residual (--use-what). Output: 128 x 128 logits (the official grid).
Data: online synthetic samples (solution/wmgen.py, no disk) + real train-fold images in every batch (flips).
Evaluation: official per-image mIoU on the real held-out fold (never trained on), by mask type, and on a fixed
synthetic validation set (gap synthetic -> real = fidelity of the generator).
Robust to interruptions: time budget, atomic checkpoints, automatic resume from <out>/last.pt.
Usage: python solution/train_seg2.py --method trustmark --device cuda --data release --out output/seg2/trustmark --minutes 60
At the end writes <out>/pred.npz (probabilities for train fold, held-out fold and test)."""
import argparse, copy, json, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "solution"))
from wmgen import Generator, Prefetch, real_items, TYPES
from common import METHODS
from wmloc.utils import miou_image


AMP_DTYPE = torch.float16                                   # --amp 2 = bfloat16 (float32 range, no NaN, no scaler needed)
AMP = True                                                   # --amp 0 for PixelSeal: its ConvNeXt-V2 extractor gives NaN in float16


class Block(nn.Module):
    def __init__(self, i, o, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(o)
        self.sk = None if (i == o and stride == 1) else nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)), inplace=True); h = self.b2(self.c2(h))
        return F.relu(h + (x if self.sk is None else self.sk(x)), inplace=True)


class SegNet(nn.Module):
    """Residual U-Net; the first stage works at full resolution (no pooling before the first convolutions)."""
    def __init__(self, cin, w=(32, 64, 128, 192, 256, 384)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, w[0], 3, 1, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True), Block(w[0], w[0]))
        self.down = nn.ModuleList([nn.Sequential(Block(w[i], w[i + 1], 2), Block(w[i + 1], w[i + 1])) for i in range(5)])   # 256,128,64,32,16
        self.up = nn.ModuleList([Block(w[i + 1] + w[i], w[i]) for i in (4, 3, 2)])                                           # 32,64,128
        self.head = nn.Conv2d(w[2], 1, 1)

    def forward(self, x):
        h = self.stem(x); feats = []
        for d in self.down:
            h = d(h); feats.append(h)
        for u, skip in zip(self.up, (feats[3], feats[2], feats[1])):
            h = u(torch.cat([F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False), skip], 1))
        return self.head(h)[:, 0]

    def run(self, items, dev): return self(make_input(items, dev, self.use_what))


class UNetSMP(nn.Module):
    """Teammate recipe (modal_t1/train_lib.py, StegaStamp 0.853 on the odd half): ImageNet ResNet-50 U-Net, 6 input
    channels = normalised image + Gaussian high-pass x 20 computed in float32, bf16 autocast, 512 logits averaged to 128."""
    def __init__(self, encoder="resnet50"):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.Unet(encoder, encoder_weights="imagenet", in_channels=6, classes=1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)); self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        ax = torch.arange(5).float() - 2; g1 = torch.exp(-ax ** 2 / 2); g1 = g1 / g1.sum()
        self.register_buffer("blur", (g1[:, None] * g1[None, :]).expand(3, 1, 5, 5).contiguous())

    def run(self, items, dev):
        x = torch.from_numpy(np.stack([it["img"] for it in items])).to(dev).permute(0, 3, 1, 2).float() / 255
        hp = (x - F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), self.blur, groups=3)) * 20.0
        inp = torch.cat([(x - self.mean) / self.std, hp], 1)
        cuda = str(dev).startswith("cuda")
        if cuda: inp = inp.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            lg = self.net(inp)
        return F.avg_pool2d(lg.float(), 4)[:, 0]


def wm_device(wm):
    d = getattr(wm, "device", "cpu")
    return d if isinstance(d, str) and (d.startswith("cuda") or d == "cpu") else "cpu"


class StegaTorch(nn.Module):
    """The 7 convolutions of the StegaStamp decoder, rebuilt in PyTorch from the ONNX weights so that they can be
    fine-tuned and give a feature pyramid (the spatial transformer and the dense head are dropped)."""
    def __init__(self, onnx_path):
        super().__init__()
        import onnx
        from onnx import numpy_helper
        g = onnx.load(str(onnx_path)).graph
        init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
        consts = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in g.node if n.op_type == "Constant"}
        self.convs = nn.ModuleList(); self.pads = []
        for n in g.node:
            if n.op_type != "Conv" or "dec_conv" not in n.name: continue
            at = {x.name: list(x.ints) for x in n.attribute if x.ints}
            w = init.get(n.input[1], consts.get(n.input[1])); b = init.get(n.input[2], consts.get(n.input[2])) if len(n.input) > 2 else None
            c = nn.Conv2d(w.shape[1], w.shape[0], w.shape[2:], stride=at.get("strides", [1, 1]), bias=b is not None)
            c.weight.data = torch.from_numpy(w.copy()).float()
            if b is not None: c.bias.data = torch.from_numpy(b.copy()).float()
            pd = at.get("pads", [0, 0, 0, 0])                 # onnx: top, left, bottom, right
            self.convs.append(c); self.pads.append((pd[1], pd[3], pd[0], pd[2]))
        assert len(self.convs) == 7, len(self.convs)

    def forward(self, x):                                    # x in [0, 1]
        h = x - 0.5; out = []
        for c, pd in zip(self.convs, self.pads):
            h = F.relu(c(F.pad(h, pd))); out.append(h)
        return [out[1], out[3], out[4], out[5], out[6]]     # strides 2, 4, 8, 16, 32


class FeatSeg(nn.Module):
    """Official extractor of the method (TrustMark ResNet-50 / PixelSeal ConvNeXt), fine-tuned, run at several input
    scales; its feature pyramid and its per-position agreement with the known message feed a segmentation head.
    (A head on the frozen TrustMark features starts learning within 200 iterations where the raw-pixel U-Net does not.)"""
    def __init__(self, wm, method, scales=(256, 512), c=64, pix=0):
        super().__init__()
        self.pix = None; self.unet = None
        if pix == 1:                                        # full-resolution image branch: object contours (masks are object segments)
            self.pix = nn.Sequential(nn.Conv2d(6, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), Block(32, 32),
                                     Block(32, 48, 2), Block(48, 64, 2), Block(64, 64))
        if pix == 2:                                        # image branch = the teammate's U-Net (ImageNet ResNet-50, image + high-pass)
            import segmentation_models_pytorch as smp
            self.unet = smp.Unet("resnet50", encoder_weights="imagenet", in_channels=6, classes=64)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)); self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            ax = torch.arange(5).float() - 2; g1 = torch.exp(-ax ** 2 / 2); g1 = g1 / g1.sum()
            self.register_buffer("blur", (g1[:, None] * g1[None, :]).expand(3, 1, 5, 5).contiguous())
        if method == "stegastamp": scales = (400, 512)
        self.method, self.scales = method, scales
        if method == "stegastamp": self.bb = StegaTorch(ROOT / "models" / "stegastamp" / "stega_stamp_dec.onnx").to(wm_device(wm))
        else: self.bb = copy.deepcopy(wm.net if method == "trustmark" else wm.det).float()
        for p in self.bb.parameters(): p.requires_grad_(True)
        with torch.no_grad():
            chans = [f.shape[1] for f in self.pyramid(torch.zeros(1, 3, 64, 64, device=next(self.bb.parameters()).device))[0]]
        self.red = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Conv2d(k, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)) for k in chans]) for _ in scales])
        cin = len(scales) * (len(chans) * c + 1) + 1
        if pix: self.body2 = nn.Sequential(Block(cin + 64, 256), Block(256, 192), Block(192, 128))
        else: self.body = nn.Sequential(Block(cin, 192), Block(192, 128))
        self.out = nn.Conv2d(128, 1, 1)

    def pyramid(self, x):
        feats = []
        if self.method == "trustmark":
            n = self.bb; h = n.maxpool(n.relu(n.bn1(n.conv1(x * 2 - 1))))
            for k in ("layer1", "layer2", "layer3", "layer4"):
                h = getattr(n, k)(h); feats.append(h)
            pp = torch.einsum("bchw,oc->bohw", h, n.fc.weight) + n.fc.bias[None, :, None, None]
        elif self.method == "stegastamp":
            feats = self.bb(x); pp = torch.zeros(len(x), 100, 1, 1, device=x.device)
        else:
            d = self.bb; h = x * 2 - 1
            for i in range(4):
                h = d.convnext.downsample_layers[i](h); h = d.convnext.stages[i](h); feats.append(h)
            emb = d.pixel_decoder.output_upscaling(h); lin = d.pixel_decoder.linear
            pp = (torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None])[:, 1:]
        return feats, pp

    def forward(self, x, ag, s):
        hs = []
        for red, sc in zip(self.red, self.scales):
            xs = x if sc == x.shape[-1] else F.interpolate(x, size=(sc, sc), mode="bilinear", align_corners=False, antialias=True)
            feats, pp = self.pyramid(xs)
            hs += [F.interpolate(r(f), size=(128, 128), mode="bilinear", align_corners=False) for r, f in zip(red, feats)]
            hs.append(F.interpolate((torch.tanh(pp.float()) * s[:, :, None, None]).mean(1, keepdim=True), size=(128, 128), mode="bilinear", align_corners=False))
        hs.append(ag[:, None, None, None].expand(-1, 1, 128, 128))
        if self.unet is not None:
            with torch.autocast("cuda", enabled=False):      # high-pass in float32: half precision erases the 1-3 grey-level trace
                xf = x.float(); hp = (xf - F.conv2d(F.pad(xf, (2, 2, 2, 2), mode="reflect"), self.blur, groups=3)) * 20.0
                inp = torch.cat([(xf - self.mean) / self.std, hp], 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                u = self.unet(inp)
            hs.append(F.relu(F.avg_pool2d(u.float(), 4)))
            return self.out(self.body2(torch.cat([h.float() for h in hs], 1)))[:, 0]
        if self.pix is not None:
            hp = x - F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, 1)
            hs.append(self.pix(torch.cat([x - 0.5, hp * 20], 1)))
            return self.out(self.body2(torch.cat([h.float() for h in hs], 1)))[:, 0]
        return self.out(self.body(torch.cat([h.float() for h in hs], 1)))[:, 0]

    def train(self, mode=True):                              # the extractor keeps its normalisation statistics
        super().train(mode); self.bb.eval(); return self

    def run(self, items, dev):
        x = torch.from_numpy(np.stack([it["img"] for it in items])).to(dev).permute(0, 3, 1, 2).float() / 255
        ag = torch.tensor([it["agree"] for it in items], device=dev, dtype=torch.float32)
        s = torch.from_numpy(np.stack([it["bits"] for it in items])).to(dev).float() * 2 - 1
        return self(x, ag, s)


def make_input(items, dev, use_what):
    x = torch.from_numpy(np.stack([it["img"] for it in items])).to(dev).permute(0, 3, 1, 2).float() / 255
    hp = x - F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, 1)
    ag = torch.tensor([it["agree"] for it in items], device=dev, dtype=torch.float32)[:, None, None, None].expand(-1, 1, 512, 512)
    oh = torch.zeros(len(items), 3, 512, 512, device=dev)
    for i, it in enumerate(items): oh[i, METHODS.index(it["method"])] = 1
    ch = [x - 0.5, hp * 20, ag, oh]
    if use_what:
        w = torch.from_numpy(np.stack([it["what"] for it in items])).to(dev).permute(0, 3, 1, 2).float()
        ch += [w / 4, hp * 255 * w / 16]
    x = torch.cat(ch, 1)
    return x.contiguous(memory_format=torch.channels_last) if str(dev).startswith("cuda") else x


def miou_loss(logit, y):
    p = torch.sigmoid(logit.float()); l = 0
    for pc, yc in ((p, y), (1 - p, 1 - y)):
        inter = (pc * yc).sum((1, 2)); union = (pc + yc - pc * yc).sum((1, 2))
        l = l + 1 - (inter + 1) / (union + 1)
    return (l / 2).mean()


def flip_item(it, k):
    if not k: return it
    img, wh, m = it["img"], it["what"], it["mask"]
    if k & 1: img, wh, m = img[:, ::-1], wh[:, ::-1], m[:, ::-1]
    if k & 2: img, wh, m = img[::-1], wh[::-1], m[::-1]
    return dict(it, img=np.ascontiguousarray(img), what=np.ascontiguousarray(wh), mask=np.ascontiguousarray(m))


@torch.no_grad()
def predict(model, items, dev, use_what, bs=16):
    model.eval(); out = []
    for i in range(0, len(items), bs):
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=AMP and dev.startswith("cuda")):
            out.append(torch.sigmoid(model.run(items[i:i + bs], dev).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 128, 128), np.float32)


def report(tag, items, probs):
    io = np.array([miou_image(p > 0.5, it["mask"]) for p, it in zip(probs, items)])
    r = {"all": float(io.mean())}
    for k in TYPES:
        sel = np.array([it["kind"] == k for it in items])
        if sel.any(): r[k] = float(io[sel].mean())
    print(f"   {tag}: " + " ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
    return r


def atomic_save(obj, path, kind="torch"):
    tmp = f"{path}.tmp{os.getpid()}"
    if kind == "torch": torch.save(obj, tmp)
    else: np.savez_compressed(tmp, **obj); tmp = tmp + ".npz"
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", required=True); ap.add_argument("--minutes", type=float, default=60); ap.add_argument("--max-steps", type=int, default=10 ** 9)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--n-real", type=int, default=4, help="real train-fold images per batch")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--use-what", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--n-synval", type=int, default=20)
    ap.add_argument("--arch", default="pix", choices=("pix", "feat", "unet")); ap.add_argument("--bb-lr", type=float, default=0.1, help="lr multiplier of the extractor")
    ap.add_argument("--probs-dir", default=""); ap.add_argument("--pipeline", default="seg2")
    ap.add_argument("--amp", type=int, default=1); ap.add_argument("--pix", type=int, default=0, help="1 = add the full-resolution image branch (--arch feat)")
    ap.add_argument("--init", default="", help="checkpoint to initialise from (tensors with matching names and shapes)")
    ap.add_argument("--gen-device", default="", help="device of the watermark models (default: --device)")
    ap.add_argument("--real-only", type=int, default=0, help="1 = no synthetic data (does the real train fold alone carry a learnable cue?)")
    ap.add_argument("--width", type=float, default=1.0); ap.add_argument("--p-real-cover", type=float, default=0.2); ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args()
    global AMP, AMP_DTYPE; AMP = bool(a.amp); AMP_DTYPE = torch.bfloat16 if a.amp == 2 else torch.float16
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = a.device; gdev = a.gen_device or dev; t_start = time.time()
    kw = dict(coco=ROOT / "extra" / "val2017", gen_dirs=[ROOT / "extra" / "diffusiondb"])
    methods = list(METHODS) if a.method == "all" else [a.method]       # "all" = one shared model for the 3 methods
    gs = [Generator(m, a.data, gdev, seed=10_000 + a.seed, p_real=0.0, **kw) for m in methods]
    print(f"{methods}: covers coco {len(gs[0].coco)} generated {len(gs[0].gen)} coco images with object masks {len(gs[0].inst)}", flush=True)
    real = [r for g in gs for r in real_items(g, a.data)]
    tr = [r for r in real if not r["val"]]; va = [r for r in real if r["val"]]
    synval = [] if a.real_only else [g.sample(k) for g in gs for k in TYPES for _ in range(a.n_synval)]
    print(f"real train fold {len(tr)}, held-out {len(va)}, synthetic val {len(synval)}, setup {time.time() - t_start:.0f}s", flush=True)
    pre = None if a.real_only else Prefetch(lambda i: Generator(methods[i % len(methods)], a.data, gdev, seed=a.seed * 100 + i, p_real=a.p_real_cover, **kw),
                                            n_threads=max(a.threads, len(methods)))
    if a.real_only: a.n_real = a.bs

    w = tuple(int(round(c * a.width / 8) * 8) for c in (32, 64, 128, 192, 256, 384))
    if a.arch == "unet":
        model = UNetSMP().to(dev)
        if dev.startswith("cuda"): model = model.to(memory_format=torch.channels_last)
        groups = [dict(params=list(model.parameters()), scale=1.0)]
    elif a.arch == "feat":
        assert len(methods) == 1, "--arch feat needs one method"
        model = FeatSeg(gs[0].wm, a.method, pix=a.pix).to(dev)
        groups = [dict(params=[p for n, p in model.named_parameters() if n.startswith("bb.")], scale=a.bb_lr),
                  dict(params=[p for n, p in model.named_parameters() if not n.startswith("bb.")], scale=1.0)]
    else:
        model = SegNet(3 + 3 + 1 + 3 + (6 if a.use_what else 0), w).to(dev); model.use_what = a.use_what
        if dev.startswith("cuda"): model = model.to(memory_format=torch.channels_last)
        groups = [dict(params=list(model.parameters()), scale=1.0)]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=a.amp == 1 and dev.startswith("cuda"))
    step, spent, hist = 0, 0.0, []
    last = out / "last.pt"
    if a.init and not last.exists():
        src = torch.load(a.init, map_location=dev)["model"]; dst = model.state_dict()
        ok = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
        model.load_state_dict(ok, strict=False); print(f"init from {a.init}: {len(ok)}/{len(dst)} tensors", flush=True)
    if last.exists():                                        # resume
        ck = torch.load(last, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step, spent, hist = ck["step"], ck["spent"], ck["hist"]
        print(f"resumed from step {step}, {spent / 60:.1f} min already spent", flush=True)
    budget = a.minutes * 60; warm = 300; t0 = time.time(); tl = 0.0; nl = 0

    def evaluate():
        r = dict(step=step, minutes=(spent + time.time() - t0) / 60)
        r["val"] = report("real held-out", va, predict(model, va, dev, a.use_what))
        if synval: r["synval"] = report("synthetic val", synval, predict(model, synval, dev, a.use_what))
        r["train"] = report("real train fold", tr[::3], predict(model, tr[::3], dev, a.use_what))
        hist.append(r)
        atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, spent=spent + time.time() - t0, hist=hist, args=vars(a)), last)
        (out / "hist.json.tmp").write_text(json.dumps(hist)); os.replace(out / "hist.json.tmp", out / "hist.json")

    while True:
        el = spent + time.time() - t0
        if el >= budget or step >= a.max_steps: break
        lr = a.lr * min(1.0, (step + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * min(1.0, el / budget))))
        for gparam in opt.param_groups: gparam["lr"] = lr * gparam.get("scale", 1.0)
        items = (pre.get(a.bs - a.n_real) if pre else []) + [flip_item(tr[j], int(rng.integers(4))) for j in rng.integers(0, len(tr), a.n_real)]
        model.train()
        y = torch.from_numpy(np.stack([it["mask"] for it in items])).to(dev).float()
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=AMP and dev.startswith("cuda")):
            logit = model.run(items, dev)
        loss = F.binary_cross_entropy_with_logits(logit.float(), y) + miou_loss(logit, y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
        step += 1; tl += loss.item(); nl += 1
        assert np.isfinite(tl), "loss is not finite"
        if step % 50 == 0:
            print(f"step {step} loss {tl / nl:.4f} lr {lr:.2e} {el / 60:.1f} min  queue {pre.q.qsize() if pre else 0} {('gen err ' + pre.err) if pre and pre.err else ''}", flush=True); tl = 0.0; nl = 0
        if step % a.eval_every == 0: evaluate()
    evaluate()
    if pre: pre.stop = True
    res = dict(names=np.array([r["name"] for r in real]), probs=predict(model, real, dev, a.use_what).astype(np.float16),
               val=np.array([r["val"] for r in real]), kind=np.array([r["kind"] for r in real]), method=np.array([r["method"] for r in real]))
    if not a.no_test:
        te = [r for g in gs for r in real_items(g, a.data, "test")]
        res.update(test_names=np.array([r["name"] for r in te]), test_probs=predict(model, te, dev, a.use_what).astype(np.float16),
                   test_agree=np.array([r["agree"] for r in te], np.float32), test_method=np.array([r["method"] for r in te]))
    atomic_save(res, out / "pred.npz", kind="npz")
    if a.probs_dir:                                          # exchange format for the ensembling bench
        pd = Path(a.probs_dir); pd.mkdir(parents=True, exist_ok=True)
        for m in methods:
            sel = res["method"] == m
            atomic_save(dict(names=res["names"][sel], probs=res["probs"][sel], used_in_training=~res["val"][sel]), pd / f"{a.pipeline}_{m}_train.npz", kind="npz")
            if not a.no_test:
                sel = res["test_method"] == m
                atomic_save(dict(names=res["test_names"][sel], probs=res["test_probs"][sel]), pd / f"{a.pipeline}_{m}_test.npz", kind="npz")
    print("wrote", out / "pred.npz", "total", f"{(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
