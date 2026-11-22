"""Do the forget images carry one of the Task-1 post-hoc watermarks (PixelSeal / TrustMark / StegaStamp)?
No reference message exists, so we compare decoder confidence mean|logit| on: forget images, clean COCO val2017 photos,
and Task-1 train images known to be watermarked with that method.
Run with the Task-1 venv from tasks/Hackathon_Watermark_Localisation_Task (cwd), arg = dir holding nothing special."""
import sys, json, numpy as np, torch
from pathlib import Path
from PIL import Image
t1 = Path.cwd(); sys.path.insert(0, str(t1))
from wmloc.watermarks import load_watermark
torch.set_num_threads(4)
un = t1.parent / "unlearning"
X = np.load(un / "forget_images.npz")["images"][::50]            # 70 images
coco = sorted((t1 / "extra/val2017").glob("*.jpg"))[:70]
meta = json.load(open(t1 / "data/train/metadata.json"))
def t(a): return torch.from_numpy(np.asarray(a)).permute(2, 0, 1).float()[None] / 255
def c224(f):
    im = Image.open(f).convert("RGB"); w, h = im.size; s = min(w, h); im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return im.resize((224, 224), Image.BICUBIC)
for name in ("trustmark", "pixelseal", "stegastamp"):
    try:
        wm = load_watermark(name, models_dir=t1 / "models", device="cpu")
        def conf(imgs):
            with torch.no_grad(): return np.array([float(wm.logits(t(i)).abs().mean()) for i in imgs])
        a = conf(list(X)); b = conf([c224(f) for f in coco])
        pos = [m["image_name"] for m in meta if m["method"] == name and m.get("full_watermark")][:25]
        c = conf([Image.open(t1 / f"data/train/images/{p}.png").convert("RGB") for p in pos])
        c2 = conf([Image.open(t1 / f"data/train/images/{p}.png").convert("RGB").resize((224, 224), Image.BICUBIC) for p in pos])
        print(f"{name}: mean|logit| forget {a.mean():.3f}±{a.std():.3f} (max {a.max():.3f}) | clean COCO@224 {b.mean():.3f}±{b.std():.3f} | watermarked {c.mean():.3f} | watermarked resized to 224 {c2.mean():.3f}", flush=True)
    except Exception as e:
        print(name, "FAILED", repr(e)[:300], flush=True)
