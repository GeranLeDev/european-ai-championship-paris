"""Name the classes absent from the forget set: for each class, take the external photos (COCO val2017) that orig
scores highest, describe them with CLIP (ImageNet-1k + LVIS-like free nouns) and dump contact sheets.
Usage: PYTHONPATH=<pkgs> python name_absent.py <dir with coco_u8.npy, coco_logits.pt, pseudo.npy>"""
import sys, numpy as np, torch, open_clip
from pathlib import Path
from PIL import Image, ImageDraw
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
out = Path(sys.argv[1]); torch.set_num_threads(4)
U = np.load(out / "coco_u8.npy", mmap_mode="r"); L = torch.load(out / "coco_logits.pt").float()
present = set(np.unique(np.load(out / "pseudo.npy")).tolist())
P = L.softmax(1); conf, pred = P.max(1)
print("COCO: n=%d, mean max-prob %.3f, frac conf>0.9: %.3f, classes predicted at least once: %d" % (len(L), conf.mean(), (conf > .9).float().mean(), len(pred.unique())))
cnt = np.bincount(pred.numpy(), minlength=100); print("argmax histogram:", cnt.tolist())
model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); tok = open_clip.get_tokenizer("ViT-B-32"); model.eval()
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1); std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
voc = list(IMAGENET_CLASSNAMES)
with torch.no_grad():
    T = torch.nn.functional.normalize(model.encode_text(tok([f"a photo of a {w}." for w in voc])), dim=-1)
K = 8; rows = []
for c in range(100):
    # rank by logit margin of class c standardised across the pool (works even when c is never the argmax)
    z = (L[:, c] - L.mean(1)) ; idx = z.topk(K).indices.numpy()
    x = torch.from_numpy(np.stack([U[i] for i in idx])).permute(0, 3, 1, 2).float() / 255
    with torch.no_grad(): f = torch.nn.functional.normalize(model.encode_image((x - mean) / std), dim=-1)
    p = (100 * f @ T.T).softmax(-1).mean(0).topk(3)
    rows.append((c, idx))
    print(f"c{c:2d} {'P' if c in present else '-'} argmax-n={cnt[c]:4d} topP={P[idx, c].mean():.2f} | " + ", ".join(f"{voc[i]}({v:.2f})" for v, i in zip(p.values, p.indices)))
s = 128
for part in range(4):
    sh = Image.new("RGB", (K * s + 44, 25 * s), "white"); d = ImageDraw.Draw(sh)
    for r, (c, idx) in enumerate(rows[part * 25:(part + 1) * 25]):
        d.text((2, r * s + 56), f"c{c}{'P' if c in present else ''}", fill="black")
        for k, i in enumerate(idx): sh.paste(Image.fromarray(np.asarray(U[i])).resize((s, s)), (44 + k * s, r * s))
    sh.save(out / f"coco_top_{part}.jpg", quality=80)
