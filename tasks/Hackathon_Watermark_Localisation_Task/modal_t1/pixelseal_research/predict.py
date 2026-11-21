"""Predict with a trained det_head run: val half (score, optional h-flip TTA) and the official test PixelSeal images.
Writes runs/<tag>/test_probs.npz (names, probs 128x128 float16, thr) and test_grids.npz (names, grids uint8, fix_grid).
    python -I predict.py <tag> [--hires 1] [--tta 1]"""
import sys; sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research")
import os; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from det_head import *

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--hires", type=int, default=0)
    ap.add_argument("--tta", type=int, default=1); ap.add_argument("--runs", default=str(Path(__file__).resolve().parent / "runs"))
    a = ap.parse_args()
    outd = Path(a.runs) / a.tag
    wm = load_wm(); dev = DEV
    feat = DetFeatures(wm.det, hires=bool(a.hires)).to(dev).eval(); head = Head(streams=1 + a.hires).to(dev).eval()
    ck = torch.load(outd / "best.pt", map_location=dev)
    sd = {(("msg.0." + k[4:]) if k.startswith("msg.") and not k[4].isdigit() or (k.startswith("msg.") and k.count(".") == 2) else k): v
          for k, v in ck["head"].items()}                    # checkpoints from before the multi-stream head
    head.load_state_dict(sd)
    if ck.get("det") is not None: feat.det.load_state_dict(ck["det"])

    @torch.no_grad()
    def predict(imgs, msgs, tta):
        out = []
        for i in range(0, len(imgs), 8):
            x = torch.from_numpy(imgs[i:i + 8]).to(dev).permute(0, 3, 1, 2).float().div(255)
            sgn = torch.from_numpy(msgs[i:i + 8]).to(dev).float() * 2 - 1
            p = torch.sigmoid(head(feat(x, sgn)))
            if tta:
                p = 0.5 * (p + torch.sigmoid(head(feat(x.flip(-1), sgn))).flip(-1))
            out.append(p.cpu().numpy())
        return np.concatenate(out)

    vmeta, vimgs, vgts, vmsgs = official("train", 1)
    for tta in sorted({0, a.tta}):
        pr = predict(vimgs, vmsgs, tta)
        sc, thr = best_thr(pr, vgts)
        print(f"val half ({len(vgts)} imgs) tta={tta}: mIoU {sc:.4f} at thr {thr:.2f} (ckpt thr {ck['thr']:.2f}: "
              f"{score([fix_grid(p, ck['thr']) for p in pr], vgts):.4f})", flush=True)
    tmeta, timgs, _, tmsgs = official("test", None)
    pr = predict(timgs, tmsgs, a.tta)
    names = np.array([e["image_name"] for e in tmeta])
    np.savez_compressed(outd / "test_probs.npz", names=names, probs=pr.astype(np.float16), thr=thr)
    np.savez_compressed(outd / "test_grids.npz", names=names, grids=np.stack([fix_grid(p, thr) for p in pr]))
    print(f"test: {len(names)} PixelSeal images -> {outd / 'test_probs.npz'} and test_grids.npz (thr {thr:.2f})")
