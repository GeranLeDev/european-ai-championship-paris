import sys, json
sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/sprintml/starter_kit")
import numpy as np, torch
from pathlib import Path
from PIL import Image
from wmloc.watermarks import load_watermark
wm = load_watermark("trustmark", Path("/Users/amil/Downloads/hackathon-watermark/sprintml/models"), "cpu"); tm = wm.tm
src = sorted(Path("/Users/amil/Downloads/val2017").glob("*.jpg"))[:3]
rng = np.random.default_rng(0)
for p in src:
    im = Image.open(p).convert("RGB").resize((512, 512))
    msg = rng.integers(0, 2, 100)
    w = np.asarray(wm.embed(np.asarray(im), msg), np.float32); c0 = np.asarray(im, np.float32)
    true_r = (w - c0)
    def t256(a): return torch.from_numpy(np.asarray(Image.fromarray(a.astype(np.uint8)).resize((256, 256), Image.BILINEAR), np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    with torch.no_grad():
        W = t256(w); C = t256(c0)
        rm = tm.removal(W).clamp(-1, 1) - W
        st, _ = tm.encoder(W, torch.from_numpy(msg[None].astype(np.float32))); r_re = st.clamp(-1, 1) - W; r_re = r_re - r_re.mean((2, 3), keepdim=True)
        st2, _ = tm.encoder(C, torch.from_numpy(msg[None].astype(np.float32))); r_c = st2.clamp(-1, 1) - C; r_c = r_c - r_c.mean((2, 3), keepdim=True)
    tr = (W - C)
    cc = lambda a, b: float(np.corrcoef(a.numpy().ravel(), b.numpy().ravel())[0, 1])
    print(p.name, "std true", float(tr.std()) * 127.5, "| corr(true, -rm)", cc(tr, -rm), "| corr(true, re-embed of W)", cc(tr, r_re), "| corr(true, embed of clean)", cc(tr, r_c),
          "| rm std", float(rm.std()) * 127.5, "rm std on clean", float((tm.removal(C).clamp(-1, 1) - C).std()) * 127.5)
