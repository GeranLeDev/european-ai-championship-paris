"""Track sam_shapes, step 2: training data from the EVEN half of the official train images, with object zones that
follow each image's own SAM masks (volume eaic-track-sam). CPU only. Output: eaic-gen /gen/<method><sub>/shard_XXX.npz
(same keys as app_data.gen_official).

    modal run tracks/sam_shapes/app_gen.py --methods trustmark --shards 50 --per 500 --sub _offsam
"""
import json
import time
from pathlib import Path

import modal

T1 = Path("/Users/amil/Downloads/hackathon-watermark/modal_t1")
HERE = T1 / "tracks" / "sam_shapes"
OFFICIAL = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release/train")
app = modal.App("eaic-track-sam-gen")
gen_vol = modal.Volume.from_name("eaic-gen", create_if_missing=True)
sam_vol = modal.Volume.from_name("eaic-track-sam", create_if_missing=True)
METHOD_ID = {"pixelseal": 0, "trustmark": 1, "stegastamp": 2}

image = (                                                     # same layers as app_data.image (cached build)
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "numpy", "pillow", "onnxruntime-gpu==1.23.2", "onnx",
        "trustmark==0.8.0", "omegaconf", "einops", "timm", "opencv-python-headless", "scipy", "safetensors",
        "requests", "pyyaml", "av", "decord", "pycocotools", "tqdm",
    )
    .add_local_dir(T1 / "kit", "/kit", copy=True)
    .run_commands(
        "cd /kit && python download_models.py --models_dir /models",
        "python -c \"from trustmark import TrustMark; "
        "TrustMark(use_ECC=False, verbose=False, secret_len=100, model_type='Q', device='cpu')\"",
    )
    .apt_install("curl", "unzip")
    .add_local_file(T1 / "gen_lib.py", "/kit/gen_lib.py")
    .add_local_dir(OFFICIAL, "/official/train")
    .add_local_file(HERE / "gen_sam.py", "/kit/gen_sam.py")
)


@app.function(image=image, volumes={"/gen": gen_vol, "/sam": sam_vol}, cpu=8, memory=16384, timeout=3 * 3600)
def gen(method: str, k: int, per: int, sub: str, p_own: float = 0.85, lo: float = 0.003, hi: float = 0.45) -> str:
    import sys

    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, "/kit")
    import gen_sam
    from wmloc.utils import downsample_mask
    from wmloc.watermarks import load_watermark

    out = Path(f"/gen/{method}{sub}/shard_{k:03d}.npz")
    if out.exists():
        return f"{method}{sub} {k} already done"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    t0 = time.time()
    meta = [e for e in json.loads(Path("/official/train/metadata.json").read_text()) if int(e["image_name"], 16) % 2 == 0]
    masks = {e["image_name"]: np.asarray(Image.open(f"/official/train/masks/{e['image_name']}.png").convert("L")) > 127 for e in meta}
    transplant = [(~masks[e["image_name"]]) if e["full_watermark"] else masks[e["image_name"]] for e in meta]
    transplant = [s for s in transplant if 0.002 <= s.mean() <= 0.5]
    own = {}
    for p in sorted(Path("/sam").glob("train_*.npz")):
        z = np.load(p); names = z["names"].tolist()
        keep = (z["area"] >= lo) & (z["area"] <= hi)
        for i, n in enumerate(names):
            sel = keep & (z["img"] == i)
            own[n] = (z["bits"][sel], z["area"][sel])
    rng = np.random.default_rng(9_000_011 * (METHOD_ID[method] + 1) + k)
    wm = load_watermark(method, "/models", "cpu")
    imgs = np.zeros((per, 512, 512, 3), np.uint8); m128 = np.zeros((per, 128, 128), np.uint8)
    msgs, regimes, srcs = [], [], []
    n_own = 0
    for i in range(per):
        e = meta[int(rng.integers(len(meta)))]
        img = np.asarray(Image.open(f"/official/train/images/{e['image_name']}.png").convert("RGB")).copy()
        old = masks[e["image_name"]] if e["method"] == method else np.zeros((512, 512), bool)
        ob, oa = own.get(e["image_name"], (np.zeros((0, 512 * 512 // 8), np.uint8), np.zeros(0, np.float32)))
        n_own += len(oa) > 0
        w, mask, msg, regime = gen_sam.make_sample(wm, method, img, rng, ob, oa, transplant, p_own)
        imgs[i] = w; m128[i] = downsample_mask((mask | old).astype(np.uint8))
        msgs.append("".join(map(str, msg.tolist()))); regimes.append(regime); srcs.append(e["image_name"])
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, images=imgs, masks128=m128, messages=np.array(msgs), regimes=np.array(regimes), sources=np.array(srcs))
    tmp.replace(out)
    gen_vol.commit()
    return f"{method}{sub} {k}: {per} samples ({n_own} with own SAM masks) in {time.time() - t0:.0f} s"


@app.local_entrypoint()
def main(methods: str = "trustmark", shards: int = 50, per: int = 500, sub: str = "_offsam", p_own: float = 0.85):
    jobs = [(m, k, per, sub, p_own) for m in methods.split(",") for k in range(shards)]
    for r in gen.starmap(jobs, order_outputs=False):
        print(r, flush=True)
