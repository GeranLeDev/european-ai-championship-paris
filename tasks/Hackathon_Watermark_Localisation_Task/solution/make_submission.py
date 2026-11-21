"""Predict the test set with a trained segmenter (or the non-learned baseline) and write a valid submission.pt.
Also prints the official held-out score (and full train score) of the same predictor.
NEVER submits (no API key; SUBMIT stays False in submission_template.py).
Usage: python solution/make_submission.py --ckpt solution/ckpt/seg.pt --device mps --out solution/submission.pt
       python solution/make_submission.py --baseline --out solution/submission_baseline.pt"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, METHODS, split_train
from wmloc.utils import score


def items(split):
    meta = json.loads((ROOT / "data" / split / "metadata.json").read_text())
    f = np.load(ROOT / "solution" / "cache" / f"feat_{split}.npz")
    feat = {n: (a, p) for n, a, p in zip(f["names"].tolist(), f["agree"], f["ppl"].astype(np.float32))}
    gt = {}
    if split == "train":
        z = np.load(ROOT / "data" / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    return [dict(name=e["image_name"], method=e["method"], agree=feat[e["image_name"]][0], ppl=feat[e["image_name"]][1],
                 mask=gt.get(e["image_name"])) for e in meta]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "solution" / "ckpt" / "seg.pt")); ap.add_argument("--device", default="cpu")
    ap.add_argument("--baseline", action="store_true"); ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "solution" / "submission.pt"))
    a = ap.parse_args()
    if a.baseline:
        from baseline import predict
        params = json.load(open(ROOT / "solution" / "cache" / "baseline_params.json"))
        pred_fn = lambda its: [predict(it["agree"], it["ppl"], *params[it["method"]]) for it in its]
    else:
        from train_seg import UNet, extra_vec, predict_probs
        model = UNet().to(a.device); model.load_state_dict(torch.load(a.ckpt, map_location=a.device)); model.eval()

        def pred_fn(its, split):
            out = []
            for i in range(0, len(its), 32):
                chunk = its[i:i + 32]
                imgs = [np.asarray(Image.open(ROOT / "data" / split / "images" / f"{it['name']}.png").convert("RGB")) for it in chunk]
                probs = predict_probs(model, imgs, [extra_vec(it["method"], it["agree"]) for it in chunk], a.device)
                out += list(probs)
            return out
    run = (lambda its, split: [p.astype(np.float32) for p in pred_fn(its)]) if a.baseline else pred_fn

    tr = items("train"); val = split_train()
    tr_probs = run(tr, "train")
    thr = {m: a.thr for m in METHODS}
    if a.tune and not a.baseline:                       # per-method threshold, tuned on the train fold only
        for m in METHODS:
            t = [i for i, it in enumerate(tr) if it["method"] == m and it["name"] not in val]
            thr[m] = max(np.linspace(0.2, 0.8, 13), key=lambda h: score([(tr_probs[i] > h) for i in t], [tr[i]["mask"] for i in t]))
        print("tuned thresholds", {m: round(float(v), 2) for m, v in thr.items()})
    preds = [(p > thr[it["method"]]).astype(np.uint8) for p, it in zip(tr_probs, tr)]
    for m in METHODS:
        v = [i for i, it in enumerate(tr) if it["method"] == m and it["name"] in val]
        t = [i for i, it in enumerate(tr) if it["method"] == m and it["name"] not in val]
        print(f"{m}: held-out {score([preds[i] for i in v], [tr[i]['mask'] for i in v]):.4f}  train-fold {score([preds[i] for i in t], [tr[i]['mask'] for i in t]):.4f}")
    v = [i for i, it in enumerate(tr) if it["name"] in val]
    print(f"ALL held-out {score([preds[i] for i in v], [tr[i]['mask'] for i in v]):.4f}")

    te = items("test")
    tp = [(p > thr[it["method"]]).astype(np.uint8) for p, it in zip(run(te, "test"), te)]
    masks = torch.from_numpy(np.stack(tp)).bool()
    names = [it["name"] for it in te]
    assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
    torch.save({"image_names": names, "masks": masks}, a.out)
    chk = torch.load(a.out, weights_only=True)
    print("wrote", a.out, tuple(chk["masks"].shape), chk["masks"].dtype, f"{Path(a.out).stat().st_size / 1e6:.1f} MB",
          "mean positive frac", float(chk["masks"].float().mean()))


if __name__ == "__main__":
    main()
