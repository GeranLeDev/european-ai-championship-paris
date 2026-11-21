"""Our generated shards: per-image / per-component features on the 128 grid.
Effective regime = what the label looks like (marked fraction > 0.5 -> 'full'), plus our own regime tag."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shapes_lib as sl
REL = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release/train")
meta = {e["image_name"]: e for e in json.loads((REL / "metadata.json").read_text())}
rows = []
for f in sorted((HERE / "shards").glob("*.npz")):
    folder = f.name.split("__")[0]
    method = folder.split("_")[0]
    src_kind = "off" if "_off" in folder else "coco"
    z = np.load(f)
    M = z["masks128"].astype(bool); reg = z["regimes"]; srcs = z["sources"]
    for i in range(len(M)):
        g = M[i]; fm = float(g.mean())
        eff_full = fm > 0.5
        r = dict(method=method, src=src_kind, folder=folder, tag=str(reg[i]), full=bool(eff_full), frac_marked=fm,
                 source=str(srcs[i]), empty=bool(not g.any()), allone=bool(g.all()))
        if src_kind == "off":
            e = meta[str(srcs[i])]
            r["src_method_same"] = e["method"] == method; r["src_full"] = bool(e["full_watermark"])
        r["r128"] = sl.image_row(~g if eff_full else g)
        rows.append(r)
    print(f.name, len(M), flush=True)
(HERE / "generated_rows.json").write_text(json.dumps(rows))
print(len(rows))
