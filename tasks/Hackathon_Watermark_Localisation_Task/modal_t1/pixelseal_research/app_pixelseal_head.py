"""PixelSeal localisation on H100: segmentation head on PixelSeal's own detector (det_head.py), message-conditioned.
Trains on the existing synthetic shards of the "eaic-gen" volume (/gen/pixelseal/shard_*.npz, 25k examples),
validates on the official train PixelSeal images with int(name,16) % 2 == 1, writes test probabilities.

NOT LAUNCHED. To run:
    cd /Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research
    modal run app_pixelseal_head.py --steps 6000 --unfreeze 1500 --hires 1 --tag _v1
Results: volume "eaic-runs", /runs/pixelseal_dethead<tag>/ (best.pt, val_probs.npz, test_probs.npz).
test_probs.npz: names, probs (N x 128 x 128 float16), thr (best val threshold) -> grid = fix_grid(prob, thr)."""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
T1 = HERE.parent
DATA = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release")
app = modal.App("eaic-pixelseal-dethead")
gen_vol = modal.Volume.from_name("eaic-gen", create_if_missing=True)
runs_vol = modal.Volume.from_name("eaic-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "numpy", "pillow", "omegaconf", "einops", "timm",
        "opencv-python-headless", "scipy", "safetensors", "requests", "pyyaml", "av", "decord", "tqdm",
        "onnxruntime-gpu==1.23.2", "onnx", "trustmark==0.8.0", "pycocotools",
    )
    .add_local_dir(T1 / "kit", "/kit", copy=True)
    .run_commands("cd /kit && python download_models.py --models_dir /models")
    .add_local_file(T1 / "gen_lib.py", "/kit/gen_lib.py")
    .add_local_dir(DATA / "train" / "images", "/official/train/images")
    .add_local_file(DATA / "train" / "metadata.json", "/official/train/metadata.json")
    .add_local_file(DATA / "train" / "masks_128.npz", "/official/train/masks_128.npz")
    .add_local_dir(DATA / "test" / "images", "/official/test/images")
    .add_local_file(DATA / "test" / "metadata.json", "/official/test/metadata.json")
    .add_local_file(HERE / "common.py", "/root/ps/common.py")
    .add_local_file(HERE / "det_head.py", "/root/ps/det_head.py")
)


@app.function(image=image, gpu="H100", volumes={"/gen": gen_vol, "/runs": runs_vol}, memory=98304, cpu=16,
              timeout=6 * 3600)
def train(steps: int = 6000, unfreeze: int = 1500, tag: str = "_v1", bs: int = 32, lr: float = 1e-3,
          blr: float = 5e-5, eval_every: int = 500, max_shards: int = 0, gen_dirs: str = "/gen/pixelseal", hires: int = 1) -> str:
    import os, subprocess, sys
    env = dict(os.environ, PS_KIT="/kit", PS_MODELS="/models", PS_REL="/official", PS_GENLIB="/kit")
    gen_vol.reload()
    cmd = [sys.executable, "/root/ps/det_head.py", "--steps", str(steps), "--bs", str(bs), "--lr", str(lr),
           "--unfreeze", str(unfreeze), "--blr", str(blr), "--eval_every", str(eval_every), "--data", gen_dirs,
           "--max_shards", str(max_shards), "--hires", str(hires), "--out", "/runs", "--tag", f"pixelseal_dethead{tag}", "--test", "1"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(p.stdout[-20000:], flush=True); print(p.stderr[-5000:], flush=True)
    runs_vol.commit()
    lines = [l for l in p.stdout.splitlines() if l.startswith("== ") or l.startswith("best")]
    return "\n".join(lines[-6:])


@app.local_entrypoint()
def main(steps: int = 6000, unfreeze: int = 1500, tag: str = "_v1", bs: int = 32, lr: float = 1e-3, blr: float = 5e-5,
         eval_every: int = 500, max_shards: int = 0, gen_dirs: str = "/gen/pixelseal", hires: int = 1):
    print(train.remote(steps, unfreeze, tag, bs, lr, blr, eval_every, max_shards, gen_dirs, hires))
