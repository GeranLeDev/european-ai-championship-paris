"""Contact sheets of official masks (minority class white)."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
REL = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release/train")
rows = json.loads((HERE / "official_rows.json").read_text())
def sheet(sel, out, cols=12, t=128):
    n = len(sel); rws = (n + cols - 1) // cols
    W = Image.new("L", (cols * (t + 4), rws * (t + 4)), 90)
    for i, r in enumerate(sel):
        m = np.asarray(Image.open(REL / "masks" / f"{r['name']}.png").convert("L")) > 127
        if r["full"]: m = ~m
        im = Image.fromarray((m * 255).astype(np.uint8)).resize((t, t), Image.NEAREST)
        W.paste(im, ((i % cols) * (t + 4) + 2, (i // cols) * (t + 4) + 2))
    W.save(out)
full = [r for r in rows if r["full"]]
part_obj = [r for r in rows if not r["full"] and r["r512"]["comps"] and r["r512"]["comps"][0]["kind"] not in ("rect", "bar")]
part_rect = [r for r in rows if not r["full"] and r["r512"]["comps"] and r["r512"]["comps"][0]["kind"] in ("rect", "bar")]
sheet(full[:96], HERE / "sheet_full_holes.png")
sheet(part_obj[:96], HERE / "sheet_partial_objects.png")
sheet(part_rect[:96], HERE / "sheet_partial_rects.png")
print(len(full), len(part_obj), len(part_rect))
