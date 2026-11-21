"""Run the rectangle search (solution/rectsearch.py) on images of one method; caches best box + scores per image.
Images whose global agreement says 'full watermark' (baseline_params.json threshold + margin) are skipped.
Usage: python solution/run_search.py --method trustmark --split train --device cuda [--shard 0/2] [--grid 24,1.3,0.25,2.0]
Output: solution/cache/search_{split}_{method}[_shardK].npz"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "solution"))
from PIL import Image
from wmloc.watermarks import load_watermark
from rectsearch import Scorer, coarse_boxes, search

ap = argparse.ArgumentParser()
ap.add_argument("--method"); ap.add_argument("--split", default="train"); ap.add_argument("--device", default="cuda")
ap.add_argument("--shard", default="0/1"); ap.add_argument("--grid", default="24,1.3,0.25,2.0"); ap.add_argument("--topk", type=int, default=16)
ap.add_argument("--bs", type=int, default=256); ap.add_argument("--margin", type=float, default=0.15); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--data", default=str(ROOT / "data")); ap.add_argument("--tag", default="")
ap.add_argument("--skip-dir", default="", help="dir of search_{split}_{method}*.npz snapshots: images already there are not searched again")
a = ap.parse_args()
torch.backends.cudnn.benchmark = True
meta = [e for e in json.loads((Path(a.data) / a.split / "metadata.json").read_text()) if e["method"] == a.method]
f = np.load(ROOT / "solution" / "cache" / f"feat_{a.split}.npz"); agree = dict(zip(f["names"].tolist(), f["agree"].tolist()))
tg = json.load(open(ROOT / "solution" / "cache" / "baseline_params.json"))[a.method][0]
meta = [e for e in meta if agree[e["image_name"]] <= tg + a.margin]
if a.skip_dir:                                             # static snapshot -> shards stay consistent across processes
    done = set()
    for p in sorted(Path(a.skip_dir).glob(f"search_{a.split}_{a.method}*.npz")): done.update(np.load(p)["names"].tolist())
    meta = [e for e in meta if e["image_name"] not in done]
k, n = map(int, a.shard.split("/")); meta = meta[k::n]
if a.limit: meta = meta[:a.limit]
wm = load_watermark(a.method, ROOT / "models", a.device)
sc = Scorer(wm, a.method, bs=a.bs)
g = [float(v) for v in a.grid.split(",")]
grid = coarse_boxes(int(g[0]), g[1], g[2], g[3])
print(a.method, a.split, "images", len(meta), "grid", len(grid), flush=True)
res = dict(names=[], box=[], score=[], coarse=[])
t0 = time.time()
out = ROOT / "solution" / "cache" / f"search_{a.split}_{a.method}{a.tag}{'' if n == 1 else f'_shard{k}'}.npz"
if out.exists():                                            # resume: keep what this exact run already wrote
    z = np.load(out); res = {key: z[key].tolist() for key in res}
    print("resume:", len(res["names"]), "images already in", out.name, flush=True)
have = set(res["names"])
for i, e in enumerate(meta):
    if e["image_name"] in have: continue
    img = np.asarray(Image.open(Path(a.data) / a.split / "images" / f"{e['image_name']}.png").convert("RGB"))
    msg = np.array([int(b) for b in e["message"]]) if isinstance(e["message"], str) else np.array(e["message"])
    while True:
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=(a.device.startswith("cuda") and a.method != "stegastamp")):
                box, s, c = search(sc, img, msg, grid, topk=a.topk)
            break
        except Exception as ex:                             # out of memory -> halve the batch and retry
            if sc.bs <= 8: raise
            sc.bs //= 2; print("retry with bs", sc.bs, "after", repr(ex)[:200], flush=True)
    res["names"].append(e["image_name"]); res["box"].append(box); res["score"].append(s); res["coarse"].append(c)
    print(f"{i} {e['image_name']} score {s:.3f} coarse {c:.3f} box {np.round(box).astype(int).tolist()} {time.time() - t0:.0f}s", flush=True)
    if True:                                                # save after every image (cheap, makes kills lossless)
        np.savez(str(out) + '.tmp.npz', names=np.array(res["names"]), box=np.array(res["box"]), score=np.array(res["score"]), coarse=np.array(res["coarse"])); Path(str(out) + '.tmp.npz').replace(out)
print("done", time.time() - t0)
