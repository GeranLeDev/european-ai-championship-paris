"""Run orig on external real photos (COCO val2017 + cached Imagenette) to find what the 55 absent classes respond to.
Usage: python probe_external.py <out_dir> [n_coco]"""
import sys, numpy as np, torch, timm
from pathlib import Path
from PIL import Image
root = Path(__file__).resolve().parents[2]; out = Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
torch.set_num_threads(4)
m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)
m.load_state_dict(torch.load(root / "model.pt", map_location="cpu", weights_only=True)); m.eval()
files = sorted((root.parent / "Hackathon_Watermark_Localisation_Task/extra/val2017").glob("*.jpg"))[:n]
def load(f):
    im = Image.open(f).convert("RGB"); w, h = im.size; s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((l, t, l + 224, t + 224)))
U, Ls = [], []
with torch.no_grad():
    for i in range(0, len(files), 64):
        u = np.stack([load(f) for f in files[i:i + 64]]); U.append(u)
        x = torch.from_numpy(u).permute(0, 3, 1, 2).float() / 255
        Ls.append(m((x - 0.5) / 0.5))
        if i % 640 == 0: print(i, flush=True)
np.save(out / "coco_u8.npy", np.concatenate(U)); torch.save(torch.cat(Ls), out / "coco_logits.pt")
