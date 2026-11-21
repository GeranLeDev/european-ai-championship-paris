"""Tâche 1 : recherche de rectangle (code de l'équipe, solution/rectsearch.py) sur des H100 Modal.

Idée : dans les images « partielles » rectangulaires, la zone a été watermarkée à part puis recollée. En décodant
exactement ce rectangle, on relit le message connu à 90-100 %. On cherche donc le cadre qui relit le mieux le message.

Lancer (depuis ce dossier, avec le venv Modal) :
    modal run app_rect.py --split train --methods trustmark --shards 1 --limit 3          # petit test
    modal run app_rect.py --split test --shards 8 --grid 16,1.25,0.2,2.0 --tag _fine_t     # vrai calcul
Récupérer les résultats :
    modal volume get eaic-t1 cache ./cache
Les fichiers ont le même format et les mêmes noms que solution/cache/ de l'équipe :
search_{split}_{method}{tag}_shard{k}.npz avec names, box (x0, y0, x1, y1), score, coarse, acc_global.
Les images « complètes » (message relu à >= 85 % sur l'image entière) ne sont pas cherchées."""
import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
DATA = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release")

app = modal.App("eaic-t1-rect")
vol = modal.Volume.from_name("eaic-t1", create_if_missing=True)

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
    .add_local_file(HERE / "rectsearch.py", "/kit/rectsearch.py")
    .add_local_dir(DATA / "train" / "images", "/data/train/images")
    .add_local_file(DATA / "train" / "metadata.json", "/data/train/metadata.json")
    .add_local_dir(DATA / "test" / "images", "/data/test/images")
    .add_local_file(DATA / "test" / "metadata.json", "/data/test/metadata.json")
)


@app.function(image=image, gpu="H100", volumes={"/out": vol}, timeout=6 * 3600)
def search_shard(split: str, method: str, k: int, n: int, grid: str, tag: str, limit: int = 0,
                 full_thr: float = 0.85, topk: int = 16) -> str:
    import sys

    import numpy as np
    import torch
    try:
        import onnxruntime as ort
        ort.preload_dlls()                                   # CUDA libraries shipped with torch, for StegaStamp
    except Exception as ex:                                  # noqa: BLE001
        print("preload_dlls:", repr(ex)[:200])
    from PIL import Image
    sys.path.insert(0, "/kit")
    from rectsearch import Scorer, coarse_boxes, search
    from wmloc.watermarks import load_watermark

    torch.backends.cudnn.benchmark = True
    meta = [e for e in json.loads(Path(f"/data/{split}/metadata.json").read_text()) if e["method"] == method][k::n]
    if limit:
        meta = meta[:limit]
    wm = load_watermark(method, "/models", "cuda")
    if method == "stegastamp":
        print("providers:", wm.sess.get_providers(), flush=True)
    sc = Scorer(wm, method, bs=512)
    g = [float(v) for v in grid.split(",")]
    boxes = coarse_boxes(int(g[0]), g[1], g[2], g[3])
    out = Path(f"/out/cache/search_{split}_{method}{tag}_shard{k}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ("names", "box", "score", "coarse", "acc_global")
    res = {key: [] for key in keys}
    if out.exists():                                         # reprise : on garde ce qui est déjà fait
        z = np.load(out)
        res = {key: z[key].tolist() for key in keys}
    have = set(res["names"])
    print(f"{method} {split} shard {k}/{n}: {len(meta)} images, {len(boxes)} cadres, {len(have)} déjà faites", flush=True)
    t0, done, skipped = time.time(), 0, 0
    for e in meta:
        name = e["image_name"]
        if name in have:
            continue
        img = np.asarray(Image.open(f"/data/{split}/images/{name}.png").convert("RGB"))
        msg = np.array([int(c) for c in e["message"]])
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255)
        acc = float((wm.decode(x)[0].cpu().numpy() == msg).mean())
        if acc >= full_thr:                                  # image « complète » : pas de rectangle à chercher
            skipped += 1
            continue
        while True:
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=method != "stegastamp"):
                    box, s, c = search(sc, img, msg, boxes, topk=topk)
                break
            except torch.cuda.OutOfMemoryError:
                sc.bs //= 2
                print("mémoire pleine, lots de", sc.bs, flush=True)
        for key, v in zip(keys, (name, [float(b) for b in box], s, c, acc)):
            res[key].append(v)
        done += 1
        print(f"{name} score {s:.3f} grossier {c:.3f} cadre {np.round(box).astype(int).tolist()} "
              f"({(time.time() - t0) / done:.1f} s/image)", flush=True)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, **{key: np.array(res[key]) for key in keys})
        tmp.replace(out)
        if done % 5 == 0:
            vol.commit()
    vol.commit()
    return f"{method} {split} shard {k}: {done} cherchées, {skipped} complètes sautées, {time.time() - t0:.0f} s"


@app.local_entrypoint()
def main(split: str = "train", methods: str = "trustmark,pixelseal,stegastamp", shards: int = 4,
         grid: str = "24,1.3,0.25,2.0", tag: str = "", limit: int = 0):
    jobs = [(split, m, k, shards, grid, tag, limit) for m in methods.split(",") for k in range(shards)]
    for r in search_shard.starmap(jobs):
        print(r)
