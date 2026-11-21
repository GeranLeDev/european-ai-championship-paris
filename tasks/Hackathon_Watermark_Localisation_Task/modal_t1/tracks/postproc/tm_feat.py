"""TrustMark-specific localisation features on CPU, from Adobe's own remover network and the encoder re-run with
the KNOWN message: per image, 128x128 maps saved to tm_feat_{split}.npz."""
import sys, json, time
sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/sprintml/starter_kit")
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from wmloc.watermarks import load_watermark
torch.set_num_threads(8)
REL = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release")
split = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
wm = load_watermark("trustmark", Path("/Users/amil/Downloads/hackathon-watermark/sprintml/models"), "cpu"); tm = wm.tm
meta = [e for e in json.loads((REL / split / "metadata.json").read_text()) if e["method"] == "trustmark"]
if split == "train":
    meta = [e for e in meta if int(e["image_name"], 16) % 2 == 1]
if limit: meta = meta[:limit]
rng = np.random.default_rng(0); NR = 3


def box(x, k):
    return F.avg_pool2d(F.pad(x, (k // 2, k - 1 - k // 2, k // 2, k - 1 - k // 2), mode="reflect"), k, 1)


out = {k: [] for k in ("rm_e", "corr", "corr_rand", "enc_e", "dot", "dot_rand")}
t0 = time.time()
with torch.no_grad():
    for i, e in enumerate(meta):
        im = Image.open(REL / split / "images" / f"{e['image_name']}.png").convert("RGB")
        c = torch.from_numpy(np.asarray(im.resize((256, 256), Image.BILINEAR), np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
        rm = tm.removal(c).clamp(-1, 1) - c                                   # ~ -residual where marked
        msgs = np.concatenate([np.array([[int(b) for b in e["message"]]]), rng.integers(0, 2, (NR, 100))]).astype(np.float32)
        cb = c.expand(len(msgs), -1, -1, -1)
        st, _ = tm.encoder(cb, torch.from_numpy(msgs))
        r = st.clamp(-1, 1) - cb; r = r - r.mean((2, 3), keepdim=True)        # predicted residual per message
        a = (-rm).mean(1, keepdim=True); a = a - box(a, 9)                       # remover estimate, high-passed lum
        b = r.mean(1, keepdim=True); b = b - box(b, 9)
        k = 8
        num = box(a * b, k); den = torch.sqrt(box(a * a, k) * box(b * b, k)) + 1e-8
        rho = (num / den)[:, 0]
        dot = box(a * b, k)[:, 0]
        to128 = lambda x: F.avg_pool2d(x[None], 2)[0] if x.dim() == 2 else F.avg_pool2d(x, 2)
        out["rm_e"].append(to128(box(rm.pow(2).mean(1, keepdim=True), 8))[0, 0].numpy())
        out["enc_e"].append(to128(box(r[:1].pow(2).mean(1, keepdim=True), 8))[0, 0].numpy())
        out["corr"].append(to128(rho[:1][None])[0, 0].numpy()); out["corr_rand"].append(to128(rho[1:][None])[0].mean(0).numpy())
        out["dot"].append(to128(dot[:1][None])[0, 0].numpy()); out["dot_rand"].append(to128(dot[1:][None])[0].mean(0).numpy())
        if i % 20 == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
np.savez_compressed(Path(__file__).resolve().parent / f"tm_feat_{split}{'_lim' if limit else ''}.npz", names=np.array([e["image_name"] for e in meta]),
                    **{k: np.stack(v).astype(np.float32) for k, v in out.items()})
print("done", len(meta), f"{time.time() - t0:.0f}s")
