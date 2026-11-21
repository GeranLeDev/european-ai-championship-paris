"""Track sam_shapes, step 1: Segment Anything (ViT-H, Apache 2.0) automatic masks on the official images.

Writes to volume eaic-track-sam: /sam/{split}_{part}_{chunk:03d}.npz with
  img (int32 [n_masks]) index into names, names (str [n_images]), bits (uint8 packbits [n_masks, 512*512/8]),
  area (float32 fraction), piou, stab (float32), bbox (float32 x,y,w,h).
Run (2 GPU containers max):
    modal run tracks/sam_shapes/app_sam.py --parts 2
"""
import json
import time
from pathlib import Path

import modal

DATA = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release")
app = modal.App("eaic-track-sam")
sam_vol = modal.Volume.from_name("eaic-track-sam", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy", "pillow", "opencv-python-headless",
                 "git+https://github.com/facebookresearch/segment-anything.git")
    .run_commands("mkdir -p /ckpt && curl -sS -L -o /ckpt/sam_vit_h.pth "
                  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
    .add_local_dir(DATA / "train" / "images", "/official/train/images")
    .add_local_file(DATA / "train" / "metadata.json", "/official/train/metadata.json")
    .add_local_dir(DATA / "test" / "images", "/official/test/images")
    .add_local_file(DATA / "test" / "metadata.json", "/official/test/metadata.json")
)


def _run(part: int, parts: int, splits: str, chunk: int, device: str) -> str:
    import numpy as np
    import torch
    from PIL import Image
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    if device == "cpu":
        torch.set_num_threads(8)
    sam = sam_model_registry["vit_h"](checkpoint="/ckpt/sam_vit_h.pth").to(device)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32, points_per_batch=256, pred_iou_thresh=0.70,
                                    stability_score_thresh=0.85, min_mask_region_area=0)
    done = 0
    for split in splits.split(","):
        meta = json.loads(Path(f"/official/{split}/metadata.json").read_text())
        names = sorted(e["image_name"] for e in meta)[part::parts]
        for c in range(0, len(names), chunk):
            out = Path(f"/sam/{split}_{parts}p{part:03d}_{c // chunk:03d}.npz")
            if out.exists():
                continue
            cn = names[c:c + chunk]
            I, B, A, P, S, BB = [], [np.zeros((0, 512 * 512 // 8), np.uint8)], [], [], [], []
            for i, n in enumerate(cn):
                img = np.asarray(Image.open(f"/official/{split}/images/{n}.png").convert("RGB"))
                with torch.inference_mode():
                    ms = gen.generate(img)
                for m in ms:
                    a = m["area"] / (512 * 512)
                    if not 0.001 <= a <= 0.6:
                        continue
                    I.append(i); B.append(np.packbits(m["segmentation"].astype(bool).ravel()))
                    A.append(a); P.append(m["predicted_iou"]); S.append(m["stability_score"]); BB.append(m["bbox"])
                done += 1
            tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez_compressed(tmp, img=np.array(I, np.int32), names=np.array(cn), bits=np.concatenate([b.reshape(-1, 512 * 512 // 8) for b in B]),
                                area=np.array(A, np.float32), piou=np.array(P, np.float32),
                                stab=np.array(S, np.float32), bbox=np.array(BB, np.float32))
            tmp.replace(out)
            sam_vol.commit()
            print(f"part {part} {split} chunk {c // chunk}: {len(cn)} images, {len(I)} masks, "
                  f"{(time.time() - t0) / max(done, 1):.2f} s/img", flush=True)
    return f"part {part}: {done} images in {time.time() - t0:.0f} s"


@app.function(image=image, gpu="H100", volumes={"/sam": sam_vol}, memory=32768, cpu=8, timeout=3 * 3600)
def run_part(part: int, parts: int, splits: str = "train,test", chunk: int = 64) -> str:
    return _run(part, parts, splits, chunk, "cuda")


@app.function(image=image, volumes={"/sam": sam_vol}, memory=16384, cpu=8, timeout=3 * 3600)
def run_part_cpu(part: int, parts: int, splits: str = "train", chunk: int = 64) -> str:
    return _run(part, parts, splits, chunk, "cpu")


@app.local_entrypoint()
def main(parts: int = 2, splits: str = "train,test", cpu: bool = False):
    fn = run_part_cpu if cpu else run_part
    for r in fn.starmap([(p, parts, splits) for p in range(parts)], order_outputs=False):
        print(r, flush=True)
