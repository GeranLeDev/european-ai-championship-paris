"""Petit test local du générateur : 4 exemples par outil sur des photos val2017, planche de contrôle et statistiques
(part du message relue sur l'image entière, comme sur les images officielles)."""
import sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
KIT, MODELS, PHOTOS, OUT = sys.argv[1:5]
sys.path.insert(0, KIT); sys.path.insert(0, str(Path(__file__).parent))
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask
import gen_lib
torch.set_num_threads(4)
files = sorted(Path(PHOTOS).glob("*.jpg"))[100:124]
rng = np.random.default_rng(0)
tiles = []
for mi, m in enumerate(("pixelseal", "trustmark", "stegastamp")):
    wm = load_watermark(m, MODELS, device="cpu"); t0 = time.time()
    for j in range(4):
        img, tr = gen_lib.load_clean(Image.open(files[mi * 8 + j]), rng)
        out, mask, msg, regime = gen_lib.make_sample(wm, m, img, rng)
        g = downsample_mask(mask.astype(np.uint8))
        x = torch.from_numpy(out).permute(2, 0, 1)[None].float() / 255
        acc = float((wm.decode(x)[0].numpy() == msg).mean())
        print(f"{m:10s} {regime:8s} marque {mask.mean():.3f} grille {g.mean():.3f} message relu {acc:.2f}")
        a = out.astype(np.float32); a[mask] = a[mask] * 0.55 + np.array([255, 0, 0]) * 0.45
        t = Image.fromarray(a.astype(np.uint8)).resize((200, 200)); ImageDraw.Draw(t).text((4, 4), f"{m[:5]} {regime} {acc:.2f}", fill="white")
        tiles.append(t)
    print(f"  {m}: {(time.time() - t0) / 4:.1f} s par exemple sur CPU")
sheet = Image.new("RGB", (4 * 204, 3 * 204), "white")
for i, t in enumerate(tiles): sheet.paste(t, ((i % 4) * 204, (i // 4) * 204))
sheet.save(OUT); print(OUT)
