"""Error analysis of task 1 on the odd validation half. Read-only on everything outside this folder.
Usage: python -I analyze.py   (kit venv)"""
import json, re, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import cv2
from scipy import ndimage

ROOT = Path("/Users/amil/Downloads/hackathon-watermark")
T1 = ROOT / "modal_t1"; DATA = ROOT / "sprintml/release"; OUT = T1 / "diag/errors"
sys.path.insert(0, str(ROOT / "sprintml/starter_kit"))
from wmloc.utils import miou_image

METHODS = ("pixelseal", "trustmark", "stegastamp")
RUN = dict(pixelseal=T1 / "pixelseal_research/runs/hires_c", trustmark=T1 / "runs/trustmark_mix", stegastamp=T1 / "runs/stegastamp_mix")
PARAMS = dict(pixelseal=(0.20, 0.35, 0), trustmark=(0.85, 0.60, 0), stegastamp=(0.40, 9.0, 0))  # s, t, d printed by combine2.py
T_FULL = 0.80


def fix_grid(prob, thr):
    p = (prob >= thr).astype(np.uint8); k = max(1, int(0.01 * p.size))
    if p.sum() == 0: p.flat[np.argsort(prob.ravel())[-k:]] = 1
    elif p.sum() == p.size: p.flat[np.argsort(prob.ravel())[:k]] = 0
    return p


def box_mask(box, d):
    m = np.zeros((512, 512), np.float32)
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    m[max(0, y0 - d):min(512, y1 + d), max(0, x0 - d):min(512, x1 + d)] = 1
    g = (m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.uint8)
    return g if 0 < g.sum() < g.size else None


def load_rect(split):
    best = {}
    pat = re.compile(rf"search_{split}_(\w+?)_f(_shard\d+)?\.npz$")
    for p in sorted((T1 / "cache_dl").rglob(f"search_{split}_*.npz")):
        if pat.fullmatch(p.name):
            z = np.load(p)
            for n, b, s in zip(z["names"].tolist(), z["box"], z["score"]):
                if n not in best or s > best[n][1]: best[n] = (np.asarray(b, np.float32), float(s))
    return best


def predict(name, prob, accv, rect, s, t, d):
    if accv < T_FULL and name in rect and rect[name][1] >= t:
        g = box_mask(rect[name][0], d)
        if g is not None: return g, True
    return fix_grid(prob, s), False


def ds128(m):
    return (m.astype(np.float32).reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.uint8)


def shape_kind(comp):
    """comp: bool array cropped to its bbox."""
    h, w = comp.shape; ar = max(h / w, w / h)
    if comp.all(): return "bar" if ar > 4 else "rect"
    yy, xx = np.mgrid[0:h, 0:w]
    ell = ((yy - (h - 1) / 2) / (h / 2)) ** 2 + ((xx - (w - 1) / 2) / (w / 2)) ** 2 <= 1
    if (ell & comp).sum() / max(1, (ell | comp).sum()) >= 0.93: return "ellipse"
    return "object"


meta = {e["image_name"]: e for e in json.loads((DATA / "train/metadata.json").read_text())}
z = np.load(DATA / "train/masks_128.npz"); GT = dict(zip(z["names"].tolist(), z["masks"].astype(np.uint8)))
ACC = json.loads((T1 / "cache/global_train.json").read_text())
RECT = load_rect("train")
JUGE = {r["name"]: r for r in json.loads((ROOT / "task1/explore/juge_train.json").read_text())}
PROBS = {}
for m in METHODS:
    f = RUN[m] / "train_probs.npz"
    if not f.exists(): f = RUN[m] / "val_probs.npz"
    zz = np.load(f); PROBS.update(dict(zip(zz["names"].tolist(), zz["probs"].astype(np.float32))))
names = [n for n in PROBS if int(n, 16) % 2 == 1]
print("val images:", len(names), Counter(meta[n]["method"] for n in names))


def auc(prob, g):
    r = ndimage.rank_data(prob.ravel()) if hasattr(ndimage, "rank_data") else None
    from scipy.stats import rankdata
    r = rankdata(prob.ravel()); gp = g.ravel().astype(bool); n1 = gp.sum(); n0 = gp.size - n1
    if n1 == 0 or n0 == 0: return np.nan
    return (r[gp].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


S_FINE = np.round(np.arange(0.02, 0.99, 0.02), 2)
rows = []
for n in names:
    e = meta[n]; m = e["method"]; full = bool(e["full_watermark"]); g = GT[n]; prob = PROBS[n]; s, t, d = PARAMS[m]
    mk = cv2.imread(str(DATA / "train/masks" / f"{n}.png"), cv2.IMREAD_GRAYSCALE) > 127
    pred, used_rect = predict(n, prob, ACC[n], RECT, s, t, d)
    net = fix_grid(prob, s)
    pred_or, used_or = predict(n, prob, 1.0 if full else 0.0, RECT, s, t, d)  # perfect regime decision
    c = 1 if g.sum() * 2 <= g.size else 0  # minority class
    gm = g == c; pm = pred == c
    minor512 = mk if c == 1 else ~mk
    lab, k = ndimage.label(minor512); comps = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        cm = lab[sl] == i
        if cm.sum() < 64: continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        whole = np.zeros((512, 512), bool); whole[sl] = cm; cg = ds128(whole).astype(bool)
        h, w = cm.shape
        comps.append(dict(kind=shape_kind(cm), box=[y0, y1, x0, x1], area=int(cm.sum()), cells=int(cg.sum()), ar=float(max(h / w, w / h)),
                          recall=float((pm & cg).sum() / cg.sum()) if cg.sum() else np.nan,
                          recall_net=float(((net == c) & cg).sum() / cg.sum()) if cg.sum() else np.nan,
                          p_in=float(prob[cg].mean()) if cg.sum() else np.nan))
    kinds = sorted({cc["kind"] for cc in comps})
    kind = "+".join(kinds) if kinds else "none"
    inter = (gm & pm).sum(); uni = (gm | pm).sum()
    cy = lambda a: np.array(ndimage.center_of_mass(a)) if a.sum() else np.array([np.nan, np.nan])
    best_s = max(S_FINE, key=lambda q: miou_image(fix_grid(prob, q), g))
    r = dict(name=n, method=m, full=full, regime="full" if full else "partial", frac=float(g.mean()), minor_class=c, minor_cells=int(gm.sum()),
             ncomp=len(comps), kind=kind, comps=comps, acc=ACC[n], pred_partial=bool(ACC[n] < T_FULL),
             rect_score=RECT[n][1] if n in RECT else None, used_rect=bool(used_rect),
             miou=miou_image(pred, g), miou_net=miou_image(net, g), miou_regime_oracle=miou_image(pred_or, g),
             miou_best_thr=miou_image(fix_grid(prob, best_s), g), best_thr=float(best_s),
             iou_min=float(inter / uni) if uni else 1.0, iou_maj=float(((~gm) & (~pm)).sum() / ((~gm) | (~pm)).sum()),
             area_ratio=float(pm.sum() / gm.sum()), recall_min=float(inter / gm.sum()), prec_min=float(inter / pm.sum()) if pm.sum() else np.nan,
             cdist=float(np.linalg.norm(cy(gm) - cy(pm))), auc=float(auc(prob, g)),
             p_in=float(prob[g == 1].mean()), p_out=float(prob[g == 0].mean()), pred_cells=int(pm.sum()))
    if n in RECT:
        rg = box_mask(RECT[n][0], 0)
        r["miou_rect"] = miou_image(rg, g) if rg is not None else None
    # border offsets for single exact-rectangle zones (cells): pred bbox of the minority component with max overlap vs true bbox
    if len(comps) == 1 and comps[0]["kind"] in ("rect", "bar") and pm.sum():
        pl, _ = ndimage.label(pm); ov = np.bincount(pl[gm], minlength=pl.max() + 1); ov[0] = 0
        if ov.max() > 0:
            ys, xs = np.where(pl == ov.argmax()); gy, gx = np.where(gm)
            if len(gy):
                r["edge_off"] = [int(ys.min() - gy.min()), int(ys.max() - gy.max()), int(xs.min() - gx.min()), int(xs.max() - gx.max())]
                r["pred_ncomp"] = int(pl.max())
    rows.append(r)

(OUT / "per_image.json").write_text(json.dumps(rows, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o))
NM = Counter(r["method"] for r in rows)
W = {r["name"]: 1 / 3 / NM[r["method"]] for r in rows}  # method-balanced share (test set is 1/3 per method)


def overall(key):
    return float(np.mean([np.mean([r[key] for r in rows if r["method"] == m]) for m in METHODS]))


print("\n== SCORES (params fitted on all val images, as combine2.py final params)")
for m in METHODS:
    rr = [r for r in rows if r["method"] == m]
    print(f"{m:10s} n {len(rr):3d} | combined {np.mean([r['miou'] for r in rr]):.4f} | net only {np.mean([r['miou_net'] for r in rr]):.4f} | "
          f"full {np.mean([r['miou'] for r in rr if r['full']]):.4f} | partial {np.mean([r['miou'] for r in rr if not r['full']]):.4f} | rect used {sum(r['used_rect'] for r in rr)}")
print(f"overall (method-balanced) {overall('miou'):.4f}   net only {overall('miou_net'):.4f}")


def bucket_area(c):
    return "<100" if c < 100 else "100-400" if c < 400 else "400-1600" if c < 1600 else ">1600"


def table(title, keyf, sub=None, top=None):
    agg = defaultdict(list)
    for r in rows:
        if sub is None or sub(r): agg[keyf(r)].append(r)
    out = []
    for k, rr in agg.items():
        lost = sum(W[r["name"]] * (1 - r["miou"]) for r in rr)
        out.append((lost, k, len(rr), np.mean([r["miou"] for r in rr]), sum(W[r["name"]] for r in rr)))
    out.sort(reverse=True)
    print(f"\n== {title}\n{'bucket':58s} {'n':>4s} {'share':>6s} {'mIoU':>6s} {'pts lost':>8s}")
    for lost, k, nn, mi, sh in (out[:top] if top else out):
        print(f"{str(k):58s} {nn:4d} {sh:6.3f} {mi:6.3f} {100 * lost:8.2f}")
    return out


table("BY METHOD x REGIME", lambda r: (r["method"], r["regime"]))
table("BY METHOD x REGIME x KIND", lambda r: (r["method"], r["regime"], r["kind"]))
table("BY METHOD x REGIME x MINORITY-AREA", lambda r: (r["method"], r["regime"], bucket_area(r["minor_cells"])))
table("BY METHOD x REGIME x N COMPONENTS", lambda r: (r["method"], r["regime"], min(r["ncomp"], 3)))
table("BY REGIME x KIND (all methods)", lambda r: (r["regime"], r["kind"]))
table("BY REGIME x MINORITY-AREA (all methods)", lambda r: (r["regime"], bucket_area(r["minor_cells"])))
full_tab = table("FULL CROSS: METHOD x REGIME x KIND x AREA x NCOMP (top 30)", lambda r: (r["method"], r["regime"], r["kind"], bucket_area(r["minor_cells"]), min(r["ncomp"], 3)), top=30)

print("\n== REGIME DECISION (global bit accuracy < 0.80 = partial)")
for m in METHODS:
    rr = [r for r in rows if r["method"] == m]
    fp = [r for r in rr if r["full"] and r["pred_partial"]]; fn = [r for r in rr if not r["full"] and not r["pred_partial"]]
    print(f"{m:10s} full called partial {len(fp)}/{sum(r['full'] for r in rr)} | partial called full {len(fn)}/{sum(not r['full'] for r in rr)}"
          f" | accs of errors: {[round(r['acc'], 2) for r in fp + fn]}")

print("\n== ORACLES (overall method-balanced; delta vs current)")
base = overall("miou")


def ov(vals):
    return float(np.mean([np.mean([v for r, v in zip(rows, vals) if r["method"] == m]) for m in METHODS]))


def per_m(vals):
    return " ".join(f"{m[:5]} {np.mean([v for r, v in zip(rows, vals) if r['method'] == m]):.3f}" for m in METHODS)


def rep(label, vals):
    print(f"{label:74s} {ov(vals):.4f}  ({100 * (ov(vals) - base):+5.2f} pts)  [{per_m(vals)}]")


rep("current", [r["miou"] for r in rows])
rep("(a) perfect regime decision, same params", [r["miou_regime_oracle"] for r in rows])
# (a2) regime-specific network threshold, true regime vs predicted regime (the second is NOT an oracle)
for lab, key in (("true", "full"), ("predicted", "pred_full")):
    vals = {}
    for m in METHODS:
        for reg in (True, False):
            rr = [r for r in rows if r["method"] == m and ((r["full"] if key == "full" else not r["pred_partial"]) == reg)]
            if not rr: continue
            best = None
            for s in S_FINE:
                sc = [r["miou"] if r["used_rect"] else miou_image(fix_grid(PROBS[r["name"]], s), GT[r["name"]]) for r in rr]
                if best is None or np.mean(sc) > best[0]: best = (np.mean(sc), s, sc)
            for r, v in zip(rr, best[2]): vals[r["name"]] = v
            print(f"      thr[{m}, {'full' if reg else 'partial'} by {lab} regime] = {best[1]:.2f} (n {len(rr)}, mIoU {best[0]:.3f})")
    rep(f"(a2) threshold per regime, {lab} regime" + (" [oracle]" if lab == "true" else " [REAL lever, fitted in-sample]"), [vals[r["name"]] for r in rows])
exact = lambda r, kinds: (not r["full"]) and r["ncomp"] >= 1 and all(c["kind"] in kinds for c in r["comps"])
rep("(b) exact GT when all true zones are rect (partial)", [1.0 if exact(r, ("rect",)) else r["miou"] for r in rows])
rep("(b') exact GT when all true zones are rect or bar (partial)", [1.0 if exact(r, ("rect", "bar")) else r["miou"] for r in rows])
rep("(b'') + full images whose holes are all exact rect/bar", [1.0 if (r["ncomp"] >= 1 and all(c["kind"] in ("rect", "bar") for c in r["comps"])) else r["miou"] for r in rows])
rep("(c) best per-image threshold on net prob (no rect)", [r["miou_best_thr"] for r in rows])
rep("(c') max(best per-image threshold, current)", [max(r["miou_best_thr"], r["miou"]) for r in rows])
rep("(d) all partial images perfect", [1.0 if not r["full"] else r["miou"] for r in rows])
rep("(e) all full images perfect", [1.0 if r["full"] else r["miou"] for r in rows])
rep("(f) partial 'object' images perfect", [1.0 if (not r["full"] and "object" in r["kind"]) else r["miou"] for r in rows])
rep("(g) rect search box used whenever it beats current (oracle pick)", [max(r["miou"], r.get("miou_rect") or 0) for r in rows])
print("   n images with all-rect zones (partial):", sum(exact(r, ("rect",)) for r in rows), "| rect or bar:", sum(exact(r, ("rect", "bar")) for r in rows))


def fail_type(r):
    if r["recall_min"] < 0.10: return "missed (recall<0.1)" if r["area_ratio"] < 2 else "missed + false zone elsewhere"
    if r["iou_min"] >= 0.75: return "ok (IoU>=0.75)"
    if r["area_ratio"] > 1.5: return "too large (>1.5x)"
    if r["area_ratio"] < 0.67: return "too small (<0.67x)"
    return "right size, misplaced/ragged"


for r in rows: r["fail"] = fail_type(r)
print("\n== FAILURE TYPE of the minority class, BY METHOD x REGIME (n, mean mIoU, pts lost)")
agg = defaultdict(list)
for r in rows: agg[(r["method"], r["regime"], r["fail"])].append(r)
for k in sorted(agg, key=lambda k: -sum(W[r["name"]] * (1 - r["miou"]) for r in agg[k])):
    rr = agg[k]
    print(f"{str(k):62s} n {len(rr):3d} mIoU {np.mean([r['miou'] for r in rr]):.3f} lost {100 * sum(W[r['name']] * (1 - r['miou']) for r in rr):5.2f}"
          f" | med AUC {np.nanmedian([r['auc'] for r in rr]):.3f} med area ratio {np.median([r['area_ratio'] for r in rr]):.2f} med cdist {np.nanmedian([r['cdist'] for r in rr]):.1f}")

print("\n== 3 COSTLIEST (regime, kind) BUCKETS PER METHOD: measured failure profile")
for m in METHODS:
    agg = defaultdict(list)
    for r in rows:
        if r["method"] == m: agg[(r["regime"], r["kind"])].append(r)
    for k in sorted(agg, key=lambda k: -sum(W[r["name"]] * (1 - r["miou"]) for r in agg[k]))[:3]:
        rr = agg[k]; q = lambda key: np.nanpercentile([r[key] for r in rr], [25, 50, 75]).round(2).tolist()
        print(f"\n{m} {k}: n {len(rr)}, mIoU {np.mean([r['miou'] for r in rr]):.3f}, lost {100 * sum(W[r['name']] * (1 - r['miou']) for r in rr):.2f} pts, rect used {sum(r['used_rect'] for r in rr)}")
        print(f"   minority IoU q25/50/75 {q('iou_min')} | recall {q('recall_min')} | precision {q('prec_min')} | pred/true area {q('area_ratio')} | centroid dist (cells) {q('cdist')}")
        print(f"   pixel AUC of prob vs GT {q('auc')} | mean prob in marked {q('p_in')} / in clean {q('p_out')} | true minority cells {q('minor_cells')} | best per-image thr {q('best_thr')} -> mIoU {np.mean([r['miou_best_thr'] for r in rr]):.3f}")
        print("   failure types:", dict(Counter(r["fail"] for r in rr).most_common()))
        eo = [r["edge_off"] for r in rr if "edge_off" in r and r["iou_min"] > 0.3]
        if eo:
            a = np.abs(np.array(eo)); print(f"   single exact-rect zones found (n {len(eo)}): |edge offset| cells mean {a.mean():.2f}, median {np.median(a):.1f}, share of edges exact {np.mean(a == 0):.2f}, off by >=2 {np.mean(a >= 2):.2f}")

print("\n== PER-ZONE RECALL (each true zone/hole; recall = share of its cells predicted as minority)")
agg = defaultdict(list)
for r in rows:
    for c in r["comps"]:
        if c["cells"] > 0: agg[(r["method"], r["regime"], c["kind"], bucket_area(c["cells"]))].append(c)
print(f"{'method, regime, zone kind, zone cells':52s} {'n':>4s} {'med recall':>10s} {'missed(<0.1)':>12s} {'found(>0.7)':>11s} {'mean p_in':>9s}")
for k in sorted(agg):
    cc = agg[k]; rc = np.array([c["recall"] for c in cc])
    print(f"{str(k):52s} {len(cc):4d} {np.median(rc):10.2f} {np.mean(rc < 0.1):12.2f} {np.mean(rc > 0.7):11.2f} {np.mean([c['p_in'] for c in cc]):9.2f}")
vanish = sum(1 for r in rows for c in r["comps"] if c["cells"] == 0)
print("zones >=64 px that vanish on the 128 grid:", vanish)

print("\n== RECT SEARCH on partial images whose zones are all rect/bar: score vs correctness")
for m in METHODS:
    rr = [r for r in rows if r["method"] == m and exact(r, ("rect", "bar")) and r.get("miou_rect") is not None]
    if rr:
        print(f"{m:10s} n {len(rr)} | rect box mIoU mean {np.mean([r['miou_rect'] for r in rr]):.3f}, share >0.9: {np.mean([r['miou_rect'] > 0.9 for r in rr]):.2f} | used {sum(r['used_rect'] for r in rr)}"
              f" | current mIoU {np.mean([r['miou'] for r in rr]):.3f} | single-zone {sum(r['ncomp'] == 1 for r in rr)}")
        for r in sorted(rr, key=lambda r: r["miou"])[:6]:
            print(f"      {r['name']} kind {r['kind']} ncomp {r['ncomp']} cells {r['minor_cells']} rect score {r['rect_score']:.2f} rect mIoU {r['miou_rect']:.2f} net mIoU {r['miou_net']:.2f} used {r['used_rect']} acc {r['acc']:.2f}")

# contact sheet of the 24 worst images
from PIL import Image, ImageDraw
worst = sorted(rows, key=lambda r: r["miou"])[:24]
S = 192; pad = 4; cols = 4; cellw = 3 * S + 2 * pad + 12; cellh = S + 30
sheet = Image.new("RGB", (cols * cellw, 6 * cellh), (255, 255, 255)); dr = ImageDraw.Draw(sheet)
for i, r in enumerate(worst):
    x = (i % cols) * cellw + 6; y = (i // cols) * cellh
    im = Image.open(DATA / "train/images" / f"{r['name']}.png").convert("RGB").resize((S, S))
    g = Image.fromarray((GT[r["name"]] * 255).astype(np.uint8)).resize((S, S), Image.NEAREST).convert("RGB")
    p = Image.fromarray((np.clip(PROBS[r["name"]], 0, 1) * 255).astype(np.uint8)).resize((S, S), Image.NEAREST).convert("RGB")
    sheet.paste(im, (x, y + 26)); sheet.paste(g, (x + S + pad, y + 26)); sheet.paste(p, (x + 2 * (S + pad), y + 26))
    dr.text((x, y + 2), f"{r['name']} {r['method']} {r['regime']} {r['kind']} n={r['ncomp']} cells={r['minor_cells']}", fill=(0, 0, 0))
    dr.text((x, y + 13), f"mIoU {r['miou']:.2f} thr {PARAMS[r['method']][0]} acc {r['acc']:.2f} {'RECT' if r['used_rect'] else 'net'} AUC {r['auc']:.2f} | image / true grid / prob", fill=(180, 0, 0))
sheet.save(OUT / "worst24.png")
print("\ncontact sheet:", OUT / "worst24.png")
print("worst 24:", [(r["name"], r["method"][:5], r["regime"], r["kind"], round(r["miou"], 2)) for r in worst])
