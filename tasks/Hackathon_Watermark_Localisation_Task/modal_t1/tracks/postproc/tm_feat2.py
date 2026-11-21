"""TrustMark matched filter on CPU: re-run the encoder on the observed image with the KNOWN message (predicts the
residual, corr 0.8 with the truth on synthetic checks) and measure its local dot product with the observed image's
high-pass at 512, against random-message nulls. 128x128 maps -> tm_mf_{split}.npz."""
import sys, json, time
sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/sprintml/starter_kit")
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from wmloc.watermarks import load_watermark
torch.set_num_threads(8)
REL = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release")
split = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
wm = load_watermark("trustmark", Path("/Users/amil/Downloads/hackathon-watermark/sprintml/models"), "cpu"); tm = wm.tm
meta = [e for e in json.loads((REL / split / "metadata.json").read_text()) if e["method"] == "trustmark"]
if split == "train":
    meta = [e for e in meta if int(e["image_name"], 16) % 2 == 1]
if limit: meta = meta[:limit]
rng = np.random.default_rng(0); NR = 6


def box(x, k):
    return F.avg_pool2d(F.pad(x, (k // 2, k - 1 - k // 2, k // 2, k - 1 - k // 2), mode="reflect"), k, 1)


def gblur(x, s):
    k = int(2 * round(3 * s) + 1); ax = torch.arange(k).float() - k // 2
    g = torch.exp(-ax ** 2 / (2 * s * s)); g = g / g.sum(); C = x.shape[1]
    y = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(y, g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)


KS = (16, 32)
out = {}
t0 = time.time()
with torch.no_grad():
    for i, e in enumerate(meta):
        im = Image.open(REL / split / "images" / f"{e['image_name']}.png").convert("RGB")
        X = torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
        c = torch.from_numpy(np.asarray(im.resize((256, 256), Image.BILINEAR), np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
        msgs = np.concatenate([np.array([[int(b) for b in e["message"]]]), rng.integers(0, 2, (NR, 100))]).astype(np.float32)
        st, _ = tm.encoder(c.expand(len(msgs), -1, -1, -1), torch.from_numpy(msgs))
        r = st.clamp(-1, 1) - c; r = r - r.mean((2, 3), keepdim=True)
        R = F.interpolate(r, size=(512, 512), mode="bilinear")                 # as TrustMark merges it
        d = R[:1] - R[1:].mean(0, keepdim=True)                                  # message-specific part
        hx = X - gblur(X, 2.0)
        hR = R - gblur(R, 2.0); hd = d - gblur(d, 2.0)
        feats = {}
        for k in KS:
            dot = box((hx * hR).sum(1, keepdim=True), k)[:, 0]                   # (1+NR) x 512 x 512
            nr = box((hR * hR).sum(1, keepdim=True), k)[:, 0] + 1e-9
            nx = box((hx * hx).sum(1, keepdim=True), k)[:, 0] + 1e-9
            z = (dot[0] - dot[1:].mean(0)) / (dot[1:].std(0) + 1e-9)
            feats[f"z{k}"] = z
            feats[f"gain{k}"] = dot[0] / nr[0]                                   # least-squares amplitude of R in x
            feats[f"rho{k}"] = dot[0] / torch.sqrt(nr[0] * nx[0])
            dd = box((hx * hd).sum(1, keepdim=True), k)[0, 0]; nd = box((hd * hd).sum(1, keepdim=True), k)[0, 0] + 1e-9
            feats[f"dgain{k}"] = dd / nd
            feats[f"drho{k}"] = dd / torch.sqrt(nd * nx[0])
        feats["lognx"] = torch.log(box((hx * hx).sum(1, keepdim=True), 16)[0, 0] + 1e-9)
        for k, v in feats.items():
            out.setdefault(k, []).append(F.avg_pool2d(v[None, None], 4)[0, 0].numpy())
        if i % 30 == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
np.savez_compressed(Path(__file__).resolve().parent / f"tm_mf_{split}{'_lim' if limit else ''}.npz", names=np.array([e["image_name"] for e in meta]),
                    **{k: np.stack(v).astype(np.float32) for k, v in out.items()})
print("done", len(meta), f"{time.time() - t0:.0f}s")
