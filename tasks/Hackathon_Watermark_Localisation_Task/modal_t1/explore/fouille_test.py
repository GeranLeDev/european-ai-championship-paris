"""Fouille des 1020 images de test : methodes, messages, doublons avec le train ou entre elles."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
root = Path(sys.argv[1]); here = Path(sys.argv[2])
te = json.loads((root / "test" / "metadata.json").read_text()); tr = json.loads((root / "train" / "metadata.json").read_text())
print("cles test:", Counter(tuple(sorted(m)) for m in te)); print("methodes test:", Counter(m["method"] for m in te))
print("messages test uniques:", len({m["message"] for m in te}), "/", len(te), "| communs avec le train:", len({m["message"] for m in te} & {m["message"] for m in tr}))
print("noms communs train/test:", len({m["image_name"] for m in te} & {m["image_name"] for m in tr}))
sig = []
for m in te:
    im = Image.open(root / "test" / "images" / f"{m['image_name']}.png")
    assert im.size == (512, 512), im.size
    v = np.asarray(im.convert("L").resize((16, 16), Image.BILINEAR), np.float32).ravel(); sig.append((v - v.mean()) / (v.std() + 1e-6))
T = np.stack(sig); S = np.load(here / "train_sig.npy"); np.save(here / "test_sig.npy", T)
C = T @ T.T / T.shape[1]; np.fill_diagonal(C, 0); X = T @ S.T / T.shape[1]
print("doublons dans le test (corr > 0.95):", int((np.triu(C) > 0.95).sum()), "| max", round(float(C.max()), 3))
print("doublons test/train (corr > 0.95):", int((X > 0.95).sum()), "| max", round(float(X.max()), 3))
i, j = np.unravel_index(C.argmax(), C.shape); print("paire la plus proche du test:", te[i]["image_name"], te[j]["image_name"])
