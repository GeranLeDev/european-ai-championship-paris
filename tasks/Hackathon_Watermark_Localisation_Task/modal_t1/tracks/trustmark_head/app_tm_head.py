"""TrustMark localisation on H100: segmentation head on TrustMark's own decoder (tm_head.py), message-conditioned.
Trains on /gen/trustmark_off (official even-half sources) + /gen/trustmark (COCO), validates on the official train
TrustMark images with int(name,16) % 2 == 1, writes test probabilities.

    cd /Users/amil/Downloads/hackathon-watermark/modal_t1/tracks/trustmark_head
    modal run --detach app_tm_head.py --tag _frozen --steps 6000
Results: volume "eaic-runs", /runs/tmhead<tag>/ (best.pt, val_probs.npz, test_probs.npz, log.txt)."""
from pathlib import Path

import modal

T1 = Path("/Users/amil/Downloads/hackathon-watermark/modal_t1")
HERE = T1 / "tracks" / "trustmark_head"
PS = T1 / "pixelseal_research"
DATA = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release")
app = modal.App("eaic-track-tmhead")
gen_vol = modal.Volume.from_name("eaic-gen", create_if_missing=True)
runs_vol = modal.Volume.from_name("eaic-runs", create_if_missing=True)

image = (   # same build steps as app_data.py (cached layers), then runtime mounts
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
    .add_local_dir(DATA / "train" / "images", "/official/train/images")
    .add_local_file(DATA / "train" / "metadata.json", "/official/train/metadata.json")
    .add_local_file(DATA / "train" / "masks_128.npz", "/official/train/masks_128.npz")
    .add_local_dir(DATA / "test" / "images", "/official/test/images")
    .add_local_file(DATA / "test" / "metadata.json", "/official/test/metadata.json")
    .add_local_file(PS / "common.py", "/root/ps/common.py")
    .add_local_file(HERE / "tm_common.py", "/root/tm/tm_common.py")
    .add_local_file(HERE / "tm_head.py", "/root/tm/tm_head.py")
)


@app.function(image=image, gpu="H100", volumes={"/gen": gen_vol, "/runs": runs_vol}, memory=98304, cpu=8,
              timeout=3 * 3600)
def train(steps: int = 6000, unfreeze: int = 0, tag: str = "_v1", bs: int = 32, lr: float = 1e-3, blr: float = 3e-5,
          eval_every: int = 500, max_shards: int = 24, gen_dirs: str = "/gen/trustmark_off,/gen/trustmark", stem: int = 1,
          os_: int = 8) -> str:
    import os, subprocess
    env = dict(os.environ, PS_KIT="/kit", PS_MODELS="/models", PS_REL="/official", PS_GENLIB="/kit")
    gen_vol.reload()
    out = f"/runs/tmhead{tag}"; os.makedirs(out, exist_ok=True)
    cmd = (f"python -u /root/tm/tm_head.py --steps {steps} --bs {bs} --lr {lr} --unfreeze {unfreeze} --blr {blr} "
           f"--eval_every {eval_every} --data {gen_dirs} --max_shards {max_shards} --stem {stem} --os {os_} "
           f"--out /runs --tag tmhead{tag} --test 1 2>&1 | grep -v 'objc\\|Warning\\|warn' | tee {out}/log.txt")
    subprocess.run(["bash", "-c", "set -o pipefail; " + cmd], env=env)
    runs_vol.commit()
    lines = [l for l in Path(f"{out}/log.txt").read_text().splitlines() if l.startswith("== ") or l.startswith("best") or "Error" in l]
    return "\n".join(lines[-14:])


@app.local_entrypoint()
def main(steps: int = 6000, unfreeze: int = 0, tag: str = "_v1", bs: int = 32, lr: float = 1e-3, blr: float = 3e-5,
         eval_every: int = 500, max_shards: int = 24, gen_dirs: str = "/gen/trustmark_off,/gen/trustmark", stem: int = 1,
         os_: int = 8):
    print(train.remote(steps, unfreeze, tag, bs, lr, blr, eval_every, max_shards, gen_dirs, stem, os_))
