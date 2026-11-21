"""python sim.py <method> <part> <nparts>  -> out/sim_<method>_<part>.json and (part 0) out/off_<method>.json
For every official PARTIAL image of the method: (a) official fingerprints, (b) the same zones re-created on a clean COCO photo
with each candidate recipe and a random message, then the same fingerprints."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/diag/recipe")
from lib import *
method, part, nparts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(2)
wm = load_watermark(method, MODELS, device="cpu")
M = [m for m in meta() if m["method"] == method and not m["full_watermark"]]
RECIPES = ["R2", "R3_ar2_32", "R3_ar2_64", "R3_sq", "R3_pad16", "R3_win256", "R4", "R5"]
out_off, out_sim = [], []; t0 = time.time()
for i, m in enumerate(M):
    if i % nparts != part: continue
    name = m["image_name"]; mask = load_mask(name); comps = comps_of(mask)
    if not comps: continue
    img = load_img(name); msg = msg_of(m)
    for r in fingerprints(wm, img, comps, msg): out_off.append(dict(name=name, **r))
    rng = np.random.default_rng(zlib.crc32(name.encode())); clean = clean_photo(i); smsg = rng.integers(0, 2, wm.nbits).astype(np.uint8)
    for rec in RECIPES:
        if rec == "R5" and len(comps) == 1: continue          # identical to R2
        sim = make(wm, clean, comps, smsg, rec)
        for r in fingerprints(wm, sim, comps, smsg): out_sim.append(dict(name=name, recipe=rec, **r))
    if (len(out_off)) % 10 == 0: print(method, part, i, len(M), f"{time.time()-t0:.0f}s", flush=True)
json.dump(out_off, open(f"out/off_{method}_{part}.json", "w")); json.dump(out_sim, open(f"out/sim_{method}_{part}.json", "w"))
print("done", method, part, f"{time.time()-t0:.0f}s")
