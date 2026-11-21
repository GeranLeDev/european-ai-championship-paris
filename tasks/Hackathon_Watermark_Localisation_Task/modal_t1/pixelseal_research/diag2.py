import sys; sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research")
import os; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from common import *
from slide_det import resize
wm = load_wm()
meta, imgs, gts, msgs = official("train", 1)
full = np.array([e["full_watermark"] for e in meta])
idx = [i for i in range(len(meta)) if full[i]][:12]
def acc(x, m):
    lg = wm.det(x)[:, 1:].float(); return ((lg > 0).float().cpu().numpy() == m[None]).mean(1)
rows = {}
with torch.no_grad():
    for i in idx:
        W = to_t(imgs[i:i+1]); m = msgs[i]
        r = {"full256": acc(resize(W,256), m).mean()}
        for s in (256, 192, 128, 96):            # crop size at 512
            cs = [W[..., y:y+s, x:x+s] for y in range(0, 513-s, s//2) for x in range(0, 513-s, s//2)]
            # only crops fully inside the gt watermarked zone (>=95%)
            g = gts[i]; keep = [k for k,(y,x) in enumerate([(y,x) for y in range(0,513-s,s//2) for x in range(0,513-s,s//2)]) if g[y//4:(y+s)//4, x//4:(x+s)//4].mean() > .95]
            if not keep: continue
            C = torch.cat([cs[k] for k in keep])
            for tgt in sorted({256, 128, max(64, s//2), max(64, s//4)}):
                r[f"c{s}->{tgt}"] = acc(resize(C, tgt) if tgt != s else C, m).mean()
        for k, v in r.items(): rows.setdefault(k, []).append(v)
for k, v in rows.items(): print(f"{k:12s} bitacc {np.mean(v):.3f}  (n={len(v)})")
