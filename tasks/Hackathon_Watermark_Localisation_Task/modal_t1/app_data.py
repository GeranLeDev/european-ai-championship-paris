"""Tâche 1 : fabrication du gros dataset sur Modal (machines sans GPU), avec les embedders exacts du kit officiel.

1. prep_coco : télécharge COCO train2017 (118k photos) + annotations dans le volume « eaic-coco », et prépare les
   contours des objets (servent de zones réalistes).
2. gen_shard : fabrique un paquet de N exemples (image marquée 512x512, grille officielle 128x128, message, régime)
   dans le volume « eaic-gen » : /gen/<outil>/shard_XXX.npz.

Lancer :
    modal run app_data.py::prep                                   # une fois
    modal run app_data.py --methods pixelseal,trustmark,stegastamp --shards 50 --per 500
"""
import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("eaic-t1-data")
coco_vol = modal.Volume.from_name("eaic-coco", create_if_missing=True)
gen_vol = modal.Volume.from_name("eaic-gen", create_if_missing=True)
METHOD_ID = {"pixelseal": 0, "trustmark": 1, "stegastamp": 2}
ZIP_SIZE = 19336861798

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "numpy", "pillow", "onnxruntime-gpu==1.23.2", "onnx",
        "trustmark==0.8.0", "omegaconf", "einops", "timm", "opencv-python-headless", "scipy", "safetensors",
        "requests", "pyyaml", "av", "decord", "pycocotools", "tqdm",
    )
    .add_local_dir(HERE / "kit", "/kit", copy=True)
    .run_commands(
        "cd /kit && python download_models.py --models_dir /models",
        "python -c \"from trustmark import TrustMark; "
        "TrustMark(use_ECC=False, verbose=False, secret_len=100, model_type='Q', device='cpu')\"",
    )
    .apt_install("curl", "unzip")
    .add_local_file(HERE / "gen_lib.py", "/kit/gen_lib.py")
)


@app.function(image=image, volumes={"/coco": coco_vol}, cpu=4, memory=24576, timeout=3 * 3600)
def prep_coco() -> str:
    import pickle
    import subprocess
    t0 = time.time()
    zp = Path("/coco/train2017.zip")
    if not zp.exists() or zp.stat().st_size != ZIP_SIZE:
        subprocess.run(["curl", "-sS", "-L", "-C", "-", "--retry", "8", "-o", str(zp),
                        "http://images.cocodataset.org/zips/train2017.zip"], check=True)
        coco_vol.commit()
    print(f"photos : {zp.stat().st_size / 1e9:.1f} Go en {time.time() - t0:.0f} s", flush=True)
    ann = Path("/coco/instances_train2017.json")
    if not ann.exists():
        subprocess.run("cd /tmp && curl -sS -L -o ann.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
                       " && unzip -o -q ann.zip annotations/instances_train2017.json"
                       " && cp annotations/instances_train2017.json /coco/", shell=True, check=True)
    idx = Path("/coco/objects.pkl")
    if not idx.exists():
        d = json.load(open(ann))
        imgs = {i["id"]: (i["file_name"], i["width"], i["height"]) for i in d["images"]}
        objs = {}
        for a in d["annotations"]:
            fn, w, h = imgs[a["image_id"]]
            if a.get("iscrowd") or not isinstance(a["segmentation"], list) or not 0.002 <= a["area"] / (w * h) <= 0.6:
                continue
            objs.setdefault(fn, []).append([[round(v, 1) for v in p] for p in a["segmentation"]])
        names = sorted(fn for fn, _, _ in imgs.values())
        with open(idx, "wb") as f:
            pickle.dump({"names": names, "objs": objs}, f)
    coco_vol.commit()
    return f"COCO prêt en {time.time() - t0:.0f} s"


@app.function(image=image, volumes={"/coco": coco_vol, "/gen": gen_vol}, cpu=8, memory=16384, timeout=3 * 3600)
def gen_shard(method: str, k: int, per: int, sub: str = "", keep_clean: bool = False) -> str:
    import pickle
    import sys
    import zipfile

    import numpy as np
    import torch
    sys.path.insert(0, "/kit")
    import gen_lib
    from wmloc.utils import downsample_mask
    from wmloc.watermarks import load_watermark

    out = Path(f"/gen/{method}{sub}/shard_{k:03d}.npz")
    if out.exists():
        return f"{method} {k} déjà fait"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    t0 = time.time()
    with open("/coco/objects.pkl", "rb") as f:
        idx = pickle.load(f)
    zf = zipfile.ZipFile("/coco/train2017.zip")
    rng = np.random.default_rng(1_000_003 * (METHOD_ID[method] + 1) + k)
    wm = load_watermark(method, "/models", "cpu")
    imgs = np.zeros((per, 512, 512, 3), np.uint8); m128 = np.zeros((per, 128, 128), np.uint8)
    cleans = np.zeros((per, 512, 512, 3), np.uint8) if keep_clean else None
    msgs, regimes, srcs = [], [], []
    names = idx["names"]
    for i in range(per):
        fn = names[int(rng.integers(len(names)))]
        img, tr = gen_lib.load_clean(zf.read(f"train2017/{fn}"), rng)
        objs = gen_lib.object_masks(idx["objs"].get(fn, []), tr)
        w, mask, msg, regime = gen_lib.make_sample(wm, method, img, rng, objs)
        imgs[i] = w; m128[i] = downsample_mask(mask.astype(np.uint8))
        if keep_clean:
            cleans[i] = img
        msgs.append("".join(map(str, msg.tolist()))); regimes.append(regime); srcs.append(fn)
    tmp = out.with_name(out.stem + ".tmp.npz")
    extra = {"cleans": cleans} if keep_clean else {}
    np.savez(tmp, images=imgs, masks128=m128, messages=np.array(msgs), regimes=np.array(regimes), sources=np.array(srcs), **extra)
    tmp.replace(out)
    gen_vol.commit()
    return f"{method} {k}: {per} exemples en {time.time() - t0:.0f} s"


OFFICIAL = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release/train")
image_off = image.add_local_dir(OFFICIAL, "/official/train")


def is_source(name: str) -> bool:
    """Moitié des images officielles du train servant de matière première (l'autre moitié sert à noter)."""
    return int(name, 16) % 2 == 0


@app.function(image=image_off, volumes={"/gen": gen_vol}, cpu=8, memory=16384, timeout=3 * 3600)
def gen_official(method: str, k: int, per: int, sub: str = "_off", keep_clean: bool = False, parity: int = 0) -> str:
    """Exemples fabriqués à partir des images officielles (même style que leur test : images générées, lisses).
    Les zones déjà marquées de l'image source restent marquées : étiquette = nouvelles zones + anciennes zones
    (si l'ancienne zone vient du même outil). Formes des nouvelles zones : des vraies formes de leurs masques."""
    import sys

    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, "/kit")
    import gen_lib
    from wmloc.utils import downsample_mask
    from wmloc.watermarks import load_watermark

    out = Path(f"/gen/{method}{sub}/shard_{k:03d}.npz")
    if out.exists():
        return f"{method}{sub} {k} déjà fait"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    t0 = time.time()
    meta = [e for e in json.loads(Path("/official/train/metadata.json").read_text()) if int(e["image_name"], 16) % 2 == parity]
    masks = {e["image_name"]: np.asarray(Image.open(f"/official/train/masks/{e['image_name']}.png").convert("L")) > 127 for e in meta}
    shapes = [(~masks[e["image_name"]]) if e["full_watermark"] else masks[e["image_name"]] for e in meta]
    shapes = [s for s in shapes if 0.002 <= s.mean() <= 0.5]
    rng = np.random.default_rng(7_000_003 * (METHOD_ID[method] + 1) + k + 50_000 * parity)
    wm = load_watermark(method, "/models", "cpu")
    imgs = np.zeros((per, 512, 512, 3), np.uint8); m128 = np.zeros((per, 128, 128), np.uint8)
    cleans = np.zeros((per, 512, 512, 3), np.uint8) if keep_clean else None
    msgs, regimes, srcs = [], [], []
    for i in range(per):
        e = meta[int(rng.integers(len(meta)))]
        img = np.asarray(Image.open(f"/official/train/images/{e['image_name']}.png").convert("RGB")).copy()
        old = masks[e["image_name"]] if e["method"] == method else np.zeros((512, 512), bool)
        w, mask, msg, regime = gen_lib.make_sample(wm, method, img, rng, shapes)
        imgs[i] = w; m128[i] = downsample_mask((mask | old).astype(np.uint8))
        if keep_clean:
            cleans[i] = img
        msgs.append("".join(map(str, msg.tolist()))); regimes.append(regime); srcs.append(e["image_name"])
    tmp = out.with_name(out.stem + ".tmp.npz")
    extra = {"cleans": cleans} if keep_clean else {}
    np.savez(tmp, images=imgs, masks128=m128, messages=np.array(msgs), regimes=np.array(regimes), sources=np.array(srcs), **extra)
    tmp.replace(out)
    gen_vol.commit()
    return f"{method}{sub} {k}: {per} exemples en {time.time() - t0:.0f} s"


@app.local_entrypoint()
def official(methods: str = "pixelseal,trustmark,stegastamp", shards: int = 50, per: int = 500, sub: str = "_off",
             keep_clean: bool = False, parity: int = 0):
    jobs = [(m, k, per, sub, keep_clean, parity) for m in methods.split(",") for k in range(shards)]
    for r in gen_official.starmap(jobs, order_outputs=False):
        print(r, flush=True)


@app.local_entrypoint()
def prep():
    print(prep_coco.remote())


@app.local_entrypoint()
def main(methods: str = "pixelseal,trustmark,stegastamp", shards: int = 50, per: int = 500, start: int = 0,
         sub: str = "", keep_clean: bool = False):
    jobs = [(m, k, per, sub, keep_clean) for m in methods.split(",") for k in range(start, start + shards)]
    for r in gen_shard.starmap(jobs, order_outputs=False):
        print(r, flush=True)
