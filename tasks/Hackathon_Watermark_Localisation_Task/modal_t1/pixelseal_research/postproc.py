"""Post-processing on saved val probabilities: single threshold vs regime-specific thresholds (regime from the
detector's global agreement with the known message, no ground truth used)."""
import sys; sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research")
from common import *

run = sys.argv[1]
z = np.load(Path(__file__).resolve().parent / "runs" / run / "val_probs.npz")
probs = z["probs"].astype(np.float32); glob = z["glob"]
meta, _, gts, _ = official("train", 1)
full = np.array([e["full_watermark"] for e in meta])
assert list(z["names"]) == [e["image_name"] for e in meta]
thrs = np.arange(0.05, 0.96, 0.05)
one = max((score([fix_grid(p, t) for p in probs], gts), t) for t in thrs)
print(f"single thr: {one[0]:.4f} (thr {one[1]:.2f})")
# regime from global agreement: best split threshold (how separable is it?)
gs = np.sort(glob); cands = (gs[1:] + gs[:-1]) / 2
acc = [(((glob > c) == full).mean(), c) for c in cands]; ra, rc = max(acc)
print(f"regime from glob agreement: acc {ra:.3f} at {rc:.3f}  (full glob mean {glob[full].mean():.2f}, partial {glob[~full].mean():.2f})")
reg = glob > rc
tf = max((score([fix_grid(p, t) for p in probs[reg]], gts[reg]), t) for t in thrs)
tp = max((score([fix_grid(p, t) for p in probs[~reg]], gts[~reg]), t) for t in thrs)
preds = [fix_grid(p, tf[1] if r else tp[1]) for p, r in zip(probs, reg)]
print(f"regime-specific thr (full {tf[1]:.2f}, partial {tp[1]:.2f}): {score(preds, gts):.4f}")
print(f"  by true regime: full {score([p for p, f in zip(preds, full) if f], gts[full]):.3f}, "
      f"partial {score([p for p, f in zip(preds, full) if not f], gts[~full]):.3f}")
base = score([np.ones_like(g) if r else np.zeros_like(g) for g, r in zip(gts, reg)], gts)
print(f"regime-only baseline (same regime decision): {base:.4f}")
