"""Tâche 1 : un réseau de segmentation par outil (U-Net), entraîné sur H100 avec notre dataset fabriqué.
Le code d'entraînement est dans train_lib.py (le même sert au test local).

Lancer :
    modal run app_train.py --methods trustmark --steps 300 --tag _smoke --max-shards 2     # petit test
    modal run app_train.py --steps 12000 --tag _v1                                          # les 3 outils en parallèle
Résultats : volume « eaic-runs », /runs/<outil><tag>/ (best.pt, val.json, train_probs.npz, test_probs.npz)."""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
DATA = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release")
app = modal.App("eaic-t1-train")
gen_vol = modal.Volume.from_name("eaic-gen", create_if_missing=True)
runs_vol = modal.Volume.from_name("eaic-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy", "pillow", "segmentation-models-pytorch")
    .add_local_file(HERE / "train_lib.py", "/root/train_lib.py")
    .add_local_dir(DATA / "train" / "images", "/official/train/images")
    .add_local_file(DATA / "train" / "metadata.json", "/official/train/metadata.json")
    .add_local_file(DATA / "train" / "masks_128.npz", "/official/train/masks_128.npz")
    .add_local_dir(DATA / "test" / "images", "/official/test/images")
    .add_local_file(DATA / "test" / "metadata.json", "/official/test/metadata.json")
)


@app.function(image=image, gpu="H100", volumes={"/gen": gen_vol, "/runs": runs_vol}, memory=98304, cpu=16,
              timeout=8 * 3600)
def train(method: str, steps: int, tag: str, bs: int = 24, lr: float = 3e-4, encoder: str = "resnet50",
          max_shards: int = 0, eval_every: int = 500, gen_sub: str = "", amp_max: float = 1.0,
          val_half: bool = False, val_parity: int = -1) -> str:
    import sys
    sys.path.insert(0, "/root")
    import train_lib
    gen_vol.reload()
    return train_lib.run(method, steps, "/gen", "/official", f"/runs/{method}{tag}", device="cuda", bs=bs, lr=lr,
                         encoder=encoder, max_shards=max_shards, eval_every=eval_every, commit=runs_vol.commit,
                         log=lambda s: print(s, flush=True), gen_sub=gen_sub, amp_max=amp_max,
                         val_half=(val_parity + 1) if val_parity >= 0 else val_half)


@app.local_entrypoint()
def main(methods: str = "pixelseal,trustmark,stegastamp", steps: int = 12000, tag: str = "_v1", bs: int = 24,
         lr: float = 3e-4, encoder: str = "resnet50", max_shards: int = 0, eval_every: int = 500,
         gen_sub: str = "", amp_max: float = 1.0, val_half: bool = False, val_parity: int = -1):
    jobs = [(m, steps, tag, bs, lr, encoder, max_shards, eval_every, gen_sub, amp_max, val_half, val_parity)
            for m in methods.split(",")]
    for r in train.starmap(jobs, order_outputs=False):
        print(r, flush=True)
