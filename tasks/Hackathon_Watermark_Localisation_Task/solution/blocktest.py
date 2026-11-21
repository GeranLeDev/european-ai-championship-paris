"""Zeroth-order block test: for each window W, z = x' + 1_W * a * r(x') (r = re-embedding residual with the known
message); delta(W) = score(z) - score(x'). Adding the watermark where it is missing helps the decoder; adding it on top
of an existing one (double watermark) hurts. Per-cell map = mean delta over windows covering it (low = marked)."""
import sys; from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wmloc.utils import to_tensor


def score_fn(wm, s):
    def f(xb):                                   # xb: B x 3 x H x W in [0,1] -> B scores
        lg = wm.logits(xb).float().cpu()
        return torch.stack([(lg * s).mean(1), (torch.tanh(lg) * s).mean(1)], 1)
    return f


def block_map(wm, img, msg, size=64, stride=32, amp=1.0, bs=64, r=None, sign=1.0):
    s = torch.from_numpy(2 * msg.astype(np.float32) - 1)
    f = score_fn(wm, s)
    x = to_tensor(img)
    if r is None:
        r = torch.from_numpy(wm.embed(img, msg).astype(np.float32) - img.astype(np.float32)).permute(2, 0, 1)[None] / 255
    base = f(x)[0]
    pos = [(y, xx) for y in range(0, 512 - size + 1, stride) for xx in range(0, 512 - size + 1, stride)]
    out = []
    for i in range(0, len(pos), bs):
        zs = []
        for y, xx in pos[i:i + bs]:
            z = x.clone(); z[:, :, y:y + size, xx:xx + size] += sign * amp * r[:, :, y:y + size, xx:xx + size]
            zs.append((z.clamp(0, 1) * 255).round() / 255)
        out.append(f(torch.cat(zs)) - base)
    d = torch.cat(out).numpy()                   # n_pos x 2
    acc = np.zeros((2, 128, 128), np.float32); cnt = np.zeros((128, 128), np.float32)
    for v, (y, xx) in zip(d, pos):
        acc[:, y // 4:(y + size) // 4, xx // 4:(xx + size) // 4] += v[:, None, None]; cnt[y // 4:(y + size) // 4, xx // 4:(xx + size) // 4] += 1
    return acc / np.maximum(cnt, 1), base.numpy()


if __name__ == "__main__":
    import argparse
    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark
    from sklearn.metrics import roc_auc_score
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); ap.add_argument("--methods", default="trustmark")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--configs", default="64:32:1,32:16:1,64:32:3")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    torch.set_num_threads(8)
    ds = LocalisationDataset(ROOT / "data", "train")
    cfgs = [tuple(float(v) for v in c.split(":")) for c in a.configs.split(",")]
    for method in a.methods.split(","):
        wm = load_watermark(method, ROOT / "models", a.device)
        for full in (True, False):
            its = [it for it in (ds[i] for i in range(len(ds))) if it["method"] == method and it["full_watermark"] == full][:a.n]
            acc = {}
            for it in its:
                g = it["mask128"].reshape(-1)
                if g.min() == g.max(): continue
                for c in cfgs:
                    for sign in (1.0, -1.0):
                        mp, _ = block_map(wm, it["image"], it["message"], int(c[0]), int(c[1]), c[2], sign=sign)
                        for j, nm in enumerate(("margin", "tanh")):
                            acc.setdefault(f"{'add' if sign > 0 else 'sub'}{int(c[0])}/{int(c[1])}x{c[2]:g}_{nm}", []).append(roc_auc_score(g, (-sign) * mp[j].reshape(-1)))
            print(method, "full" if full else "partial", {k: round(float(np.mean(v)), 3) for k, v in acc.items()}, flush=True)
