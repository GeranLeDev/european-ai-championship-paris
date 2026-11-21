"""Combine la recherche de rectangle avec la décision « complète / partielle » et écrit une soumission.

Règle par image :
  - message relu sur l'image entière >= seuil_complet  -> image « complète » : tout à 1
  - sinon, image « partielle » : si le meilleur cadre trouvé a un score >= t (réglé par méthode) -> ce rectangle
    (agrandi de d pixels), sinon tout à 0.
t et d sont réglés sur le train (validation croisée en 2 moitiés pour un chiffre honnête), puis appliqués au test.

Usage (venv du kit) :
    python eval_rect.py --data <release> --models <models> --kit <starter_kit> --cache cache [--tags _fine] \
        [--test submission_rect.pt]"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--models", required=True); ap.add_argument("--kit", required=True)
ap.add_argument("--cache", default="cache"); ap.add_argument("--tags", default="", help="liste séparée par des virgules")
ap.add_argument("--test", default="", help="chemin du fichier de soumission à écrire")
a = ap.parse_args()
sys.path.insert(0, a.kit)
from wmloc.utils import miou_image  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
DATA, CACHE = Path(a.data), Path(a.cache)


def global_acc(split):
    """Part des bits du message relus sur l'image entière, pour toutes les images (mise en cache locale)."""
    f = CACHE / f"global_{split}.json"
    if f.exists():
        return json.loads(f.read_text())
    import torch
    from PIL import Image
    from wmloc.watermarks import load_watermark
    torch.set_num_threads(4)
    meta = json.loads((DATA / split / "metadata.json").read_text()); out = {}
    for m in METHODS:
        wm = load_watermark(m, a.models, device="cpu")
        for e in meta:
            if e["method"] != m:
                continue
            img = np.asarray(Image.open(DATA / split / "images" / f"{e['image_name']}.png").convert("RGB"))
            x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
            msg = np.array([int(c) for c in e["message"]])
            out[e["image_name"]] = float((wm.decode(x)[0].cpu().numpy() == msg).mean())
        print(f"lecture globale {split} {m} faite", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(out))
    return out


def load_search(split):
    """Meilleur cadre par image, toutes étiquettes et tous morceaux confondus (le meilleur score gagne)."""
    best = {}
    for tag in a.tags.split(","):
        pat = re.compile(rf"search_{split}_(\w+?){re.escape(tag)}(_shard\d+)?\.npz$")
        for p in sorted(CACHE.glob(f"search_{split}_*.npz")):
            mt = pat.fullmatch(p.name)
            if not mt or mt.group(1) not in METHODS:
                continue
            z = np.load(p)
            for n, b, s in zip(z["names"].tolist(), z["box"], z["score"]):
                if n not in best or s > best[n][1]:
                    best[n] = (np.asarray(b, np.float32), float(s))
    return best


def box_mask(box, d):
    m = np.zeros((512, 512), np.float32)
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    m[max(0, y0 - d):min(512, y1 + d), max(0, x0 - d):min(512, x1 + d)] = 1
    return (m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.uint8)


def predict(name, method, acc, search, t_full, t, d):
    if acc[name] >= t_full:
        return np.ones((128, 128), np.uint8)
    if name in search and search[name][1] >= t:
        return box_mask(search[name][0], d)
    return np.zeros((128, 128), np.uint8)


T_FULL = 0.80
T_GRID = np.round(np.arange(0.30, 0.96, 0.025), 3)
D_GRID = (-4, 0, 2, 4, 8)

meta_tr = json.loads((DATA / "train" / "metadata.json").read_text())
z = np.load(DATA / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
acc_tr, s_tr = global_acc("train"), load_search("train")
print(f"cadres trouvés sur le train : {len(s_tr)}")


def tune(items):
    best = None
    for t in T_GRID:
        for d in D_GRID:
            sc = np.mean([miou_image(predict(e["image_name"], e["method"], acc_tr, s_tr, T_FULL, t, d), gt[e["image_name"]]) for e in items])
            if best is None or sc > best[0]:
                best = (sc, t, d)
    return best


params, report = {}, []
rng = np.random.default_rng(0)
for m in METHODS:
    items = [e for e in meta_tr if e["method"] == m]
    base = np.mean([miou_image(np.ones((128, 128), np.uint8) if acc_tr[e["image_name"]] >= T_FULL else np.zeros((128, 128), np.uint8), gt[e["image_name"]]) for e in items])
    idx = rng.permutation(len(items)); halves = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
    cv = []
    for h in (0, 1):                                         # réglé sur une moitié, noté sur l'autre
        _, t, d = tune([items[i] for i in halves[h]])
        cv += [miou_image(predict(items[i]["image_name"], m, acc_tr, s_tr, T_FULL, t, d), gt[items[i]["image_name"]]) for i in halves[1 - h]]
    sc, t, d = tune(items); params[m] = (t, d)
    part = [e for e in items if acc_tr[e["image_name"]] < T_FULL]
    used = sum(1 for e in part if e["image_name"] in s_tr and s_tr[e["image_name"]][1] >= t)
    report.append((m, base, np.mean(cv), t, d, used, len(part)))
print("\nméthode     | sans rectangle | avec rectangle (validation croisée) | seuil t | marge d | rectangles utilisés")
for m, base, cv, t, d, used, n in report:
    print(f"{m:11s} | {base:.3f}          | {cv:.3f}                               | {t:.3f}   | {d:+d}      | {used}/{n}")
print(f"global      | {np.mean([r[1] for r in report]):.3f}          | {np.mean([r[2] for r in report]):.3f}")

if a.test:
    import torch
    meta_te = json.loads((DATA / "test" / "metadata.json").read_text())
    acc_te, s_te = global_acc("test"), load_search("test")
    names, masks = [], []
    for e in meta_te:
        t, d = params[e["method"]]
        p = predict(e["image_name"], e["method"], acc_te, s_te, T_FULL, t, d)
        assert p.shape == (128, 128) and p.sum() > 0 or acc_te[e["image_name"]] < T_FULL
        names.append(e["image_name"]); masks.append(p)
    used = sum(1 for e in meta_te if acc_te[e["image_name"]] < T_FULL and e["image_name"] in s_te and s_te[e["image_name"]][1] >= params[e["method"]][0])
    torch.save({"image_names": names, "masks": torch.from_numpy(np.stack(masks)).bool()}, a.test)
    print(f"\nsoumission écrite : {a.test} ({len(names)} images, {used} rectangles, {sum(acc_te[n] >= T_FULL for n in names)} complètes)")
