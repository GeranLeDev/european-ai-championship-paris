"""Regime switch: images decoded as 'full' (global bit acc >= 0.80) take the detector head (optionally averaged with
the U-Net), the others keep the U-Net. Each source is logit-shifted so its best threshold (net only, on its own
regime subset of the odd half) lands on 0.5, so combine2's single threshold works. Scores: net only and +rect (CV).

    python -I eval_switch.py HEAD_VAL.npz HEAD_TEST.npz [--write OUT_DIR --wfull 1.0]"""
import sys, json, argparse
from pathlib import Path
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
ap = argparse.ArgumentParser(); ap.add_argument("hv"); ap.add_argument("ht"); ap.add_argument("--write", default=""); ap.add_argument("--wfull", type=float, default=-1)
A = ap.parse_args(ARGS)
sys.argv = [sys.argv[0], A.hv]
src = open(Path(__file__).parent / "eval_combo.py").read()
exec(src[:src.index("U = load(a.unet)")])   # helpers only
import numpy as np
T1p = Path("/Users/amil/Downloads/hackathon-watermark/modal_t1")
acc_te = json.loads((T1p / "cache/global_test.json").read_text())
U = load(T1p / "runs/trustmark_mix/train_probs.npz"); H = load(A.hv)
TU = load(T1p / "runs/trustmark_mix/test_probs.npz"); TH = load(A.ht)
items = [e for e in meta if e["method"] == "trustmark" and e["image_name"] in H and int(e["image_name"], 16) % 2 == 1]
isfull = lambda n, acc: acc[n] >= T_FULL
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
sig = lambda x: 1 / (1 + np.exp(-x))


def best_t(P, sub):
    return max((np.mean([miou_image(fix_grid(P[e["image_name"]], s), gt[e["image_name"]]) for e in sub]), s) for s in np.round(np.arange(0.05, 0.96, 0.025), 3))


fsub = [e for e in items if isfull(e["image_name"], acc_tr)]; psub = [e for e in items if not isfull(e["image_name"], acc_tr)]
tu_p = best_t(U, psub); tu_f = best_t(U, fsub); th_f = best_t(H, fsub)
print(f"decided full {len(fsub)} partial {len(psub)} | U-Net partial {tu_p[0]:.4f}@{tu_p[1]} full {tu_f[0]:.4f}@{tu_f[1]} | head full {th_f[0]:.4f}@{th_f[1]}")
results = {}
for wf in ([A.wfull] if A.wfull >= 0 else [0.0, 0.5, 0.7, 1.0]):
    # full-decided images: wf * head + (1-wf) * U-Net, averaged in prob space after each is shifted to thr 0.5
    def build(Hd, Ud, acc, names):
        out = {}
        for n in names:
            if isfull(n, acc):
                h = sig(lg(Hd[n]) - lg(th_f[1])); u = sig(lg(Ud[n]) - lg(tu_f[1]))
                out[n] = wf * h + (1 - wf) * u
            else:
                out[n] = sig(lg(Ud[n]) - lg(tu_p[1]))
        return out
    V = build(H, U, acc_tr, [e["image_name"] for e in items])
    results[wf] = evaluate(V, f"switch wfull={wf}")
    if A.write and A.wfull >= 0:
        out = Path(A.write); out.mkdir(parents=True, exist_ok=True)
        T = build(TH, TU, acc_te, sorted(TU))
        vn = sorted(V); tn = sorted(T)
        np.savez_compressed(out / "val_probs.npz", names=np.array(vn), probs=np.stack([V[n] for n in vn]).astype(np.float16))
        np.savez_compressed(out / "test_probs.npz", names=np.array(tn), probs=np.stack([T[n] for n in tn]).astype(np.float16))
        print("written", out, len(vn), len(tn))
