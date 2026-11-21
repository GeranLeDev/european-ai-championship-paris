"""Official train set (510 images): per-image / per-component features at 512 and at 128.
Writes official_rows.json (one row per image, both resolutions)."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shapes_lib as sl

REL = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release/train")
meta = json.loads((REL / "metadata.json").read_text())
z = np.load(REL / "masks_128.npz"); g128 = dict(zip(z["names"].tolist(), z["masks"]))
rows = []
for e in meta:
    n = e["image_name"]
    m = np.asarray(Image.open(REL / "masks" / f"{n}.png").convert("L")) > 127
    full = bool(e["full_watermark"])
    mino = ~m if full else m
    r512 = sl.image_row(mino)
    g = g128[n].astype(bool)
    r128 = sl.image_row(~g if full else g)
    rows.append(dict(name=n, method=e["method"], full=full, odd=int(n, 16) % 2, r512=r512, r128=r128,
                     frac_marked=float(m.mean())))
(HERE / "official_rows.json").write_text(json.dumps(rows))
print(len(rows), "rows")
