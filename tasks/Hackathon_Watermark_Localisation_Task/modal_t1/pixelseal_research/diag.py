import sys; sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research")
import os; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from common import *
from matched_filter import hp
from gen_lib import load_clean
wm = load_wm(); rng = np.random.default_rng(1)
def c(a, b, m=None):
    a = a.flatten(); b = b.flatten()
    if m is not None: m = m.flatten(); a = a[m]; b = b[m]
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
# synthetic
for f in sorted(Path("/Users/amil/Downloads/val2017").glob("*.jpg"))[10:13]:
    img, _ = load_clean(Image.open(f), rng); msg = rng.integers(0, 2, 256)
    W = to_t(wm.embed(img, msg)[None]); I = to_t(img[None])
    Rm = embed_float(wm, W, msg[None]) - W; Rr = embed_float(wm, W, rng.integers(0,2,(1,256))) - W
    print("synth", f.name, "hpW.hpRm", round(c(hp(W), hp(Rm)),4), "hpW.hpRr", round(c(hp(W), hp(Rr)),4), "hpI.hpRm", round(c(hp(I), hp(Rm)),4), "hp(W-I).hpRm", round(c(hp(W-I), hp(Rm)),3))
meta, imgs, gts, msgs = official("train", 1)
for i in range(8):
    W = to_t(imgs[i:i+1]); m = msgs[i]
    Rm = embed_float(wm, W, m[None]) - W; Rr = embed_float(wm, W, rng.integers(0,2,(1,256))) - W
    g = torch.from_numpy(np.kron(gts[i], np.ones((4,4)))).bool()[None,None].to(W.device)
    print("off", meta[i]["full_watermark"], f"gt {gts[i].mean():.2f}", "in:", round(c(hp(W), hp(Rm), g),4), round(c(hp(W), hp(Rr), g),4), " out:", round(c(hp(W), hp(Rm), ~g),4), round(c(hp(W), hp(Rr), ~g),4),
          "dec", float((wm.decode(W).cpu().numpy()[0]==m).mean()))
