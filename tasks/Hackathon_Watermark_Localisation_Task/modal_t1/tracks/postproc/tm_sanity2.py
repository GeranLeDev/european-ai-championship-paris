import sys
sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/sprintml/starter_kit")
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from wmloc.watermarks import load_watermark

def box(x, k):
    return F.avg_pool2d(F.pad(x, (k // 2, k - 1 - k // 2, k // 2, k - 1 - k // 2), mode="reflect"), k, 1)
def gblur(x, s):
    k = int(2 * round(3 * s) + 1); ax = torch.arange(k).float() - k // 2
    g = torch.exp(-ax ** 2 / (2 * s * s)); g = g / g.sum(); C = x.shape[1]
    y = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(y, g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)

wm = load_watermark("trustmark", Path("/Users/amil/Downloads/hackathon-watermark/sprintml/models"), "cpu"); tm = wm.tm
rng = np.random.default_rng(1)
for p in sorted(Path("/Users/amil/Downloads/val2017").glob("*.jpg"))[3:7]:
    im = np.asarray(Image.open(p).convert("RGB").resize((512, 512)))
    msg = rng.integers(0, 2, 100); w = wm.embed(im, msg).copy(); w[:, 256:] = im[:, 256:]   # right half clean
    X = torch.from_numpy(w.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    c = torch.from_numpy(np.asarray(Image.fromarray(w).resize((256, 256), Image.BILINEAR), np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    msgs = np.concatenate([msg[None], rng.integers(0, 2, (6, 100))]).astype(np.float32)
    with torch.no_grad():
        st, _ = tm.encoder(c.expand(7, -1, -1, -1), torch.from_numpy(msgs))
    r = st.clamp(-1, 1) - c; r = r - r.mean((2, 3), keepdim=True); R = F.interpolate(r, size=(512, 512), mode="bilinear")
    hx = X - gblur(X, 2.0); hR = R - gblur(R, 2.0)
    true = torch.from_numpy((wm.embed(im, msg).astype(np.float32) - im) / 127.5).permute(2, 0, 1)[None]; ht = true - gblur(true, 2.0)
    dot = box((hx * hR).sum(1, keepdim=True), 32)[:, 0]; z = (dot[0] - dot[1:].mean(0)) / (dot[1:].std(0) + 1e-9)
    print(p.name, "z left(marked) %.2f right(clean) %.2f" % (z[:, :240].mean(), z[:, 272:].mean()),
          "| corr(hp true, hp R) %.2f" % np.corrcoef(ht[..., :256].numpy().ravel(), hR[:1, :, :, :256].numpy().ravel())[0, 1],
          "| std hp true %.2f, hp R %.2f, hp x %.2f (x255)" % (ht.std() * 127.5, hR[0].std() * 127.5, hx.std() * 127.5))
