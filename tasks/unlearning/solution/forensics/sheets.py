"""Contact sheets of the forget set grouped by orig pseudo-label (forensics, read-only on data)."""
import sys, numpy as np, torch
from pathlib import Path
from PIL import Image, ImageDraw
root = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
X = np.load(root / "forget_images.npz", mmap_mode=None)["images"]
L = torch.load(root / "solution/cache/orig_forget_logits.pt", map_location="cpu")
if isinstance(L, dict): L = L.get("logits", next(iter(L.values())))
y = L.float().argmax(1).numpy()
print("images", X.shape, X.dtype, "classes present", len(np.unique(y)))
cnt = np.bincount(y, minlength=100)
print("counts", {int(c): int(cnt[c]) for c in np.nonzero(cnt)[0]})
np.save(out / "pseudo.npy", y)
classes = [int(c) for c in np.nonzero(cnt)[0]]
K = 8  # images per class per row
for s in range(0, len(classes), 9):
    grp = classes[s:s + 9]
    sheet = Image.new("RGB", (K * 224 + 60, len(grp) * 224), "white")
    d = ImageDraw.Draw(sheet)
    for r, c in enumerate(grp):
        idx = np.nonzero(y == c)[0]
        d.text((4, r * 224 + 100), f"c{c}\nn={len(idx)}", fill="black")
        for k, i in enumerate(idx[:: max(1, len(idx) // K)][:K]):
            sheet.paste(Image.fromarray(X[i]), (60 + k * 224, r * 224))
    sheet.save(out / f"sheet_{s // 9}.jpg", quality=85)
