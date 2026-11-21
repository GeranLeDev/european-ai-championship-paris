"""Test local de bout en bout : un mini paquet fabriqué (format identique à Modal) puis quelques pas d'entraînement."""
import sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
KIT, MODELS, PHOTOS, OFFICIAL, WORK = sys.argv[1:6]
sys.path.insert(0, KIT); sys.path.insert(0, str(Path(__file__).parent))
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask
import gen_lib, train_lib
torch.set_num_threads(6)
m = "trustmark"; per = 48; rng = np.random.default_rng(3)
wm = load_watermark(m, MODELS, device="cpu"); files = sorted(Path(PHOTOS).glob("*.jpg"))
imgs = np.zeros((per, 512, 512, 3), np.uint8); m128 = np.zeros((per, 128, 128), np.uint8); msgs, regs = [], []
t0 = time.time()
for i in range(per):
    img, tr = gen_lib.load_clean(Image.open(files[int(rng.integers(len(files)))]), rng)
    w, mask, msg, regime = gen_lib.make_sample(wm, m, img, rng)
    imgs[i] = w; m128[i] = downsample_mask(mask.astype(np.uint8)); msgs.append("".join(map(str, msg))); regs.append(regime)
print(f"fabrication : {(time.time() - t0) / per:.2f} s par exemple")
d = Path(WORK) / "gen" / m; d.mkdir(parents=True, exist_ok=True)
np.savez(d / "shard_000.npz", images=imgs, masks128=m128, messages=np.array(msgs), regimes=np.array(regs), sources=np.array(["x"] * per))
print(train_lib.run(m, 12, Path(WORK) / "gen", OFFICIAL, Path(WORK) / "runs" / m, device="mps", bs=4, encoder="resnet34",
                    eval_every=6, pretrained=False, val_limit=8))
z = np.load(Path(WORK) / "runs" / m / "test_probs.npz"); print("probas test :", z["probs"].shape, z["names"][:2])
