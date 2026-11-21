"""Entraînement d'un réseau de segmentation par outil (tâche 1). Utilisé par app_train.py (H100) et en local (test).

Entrée : image 512x512. Sortie : grille 128x128 (probabilité « watermarké » par case de 4x4 pixels).
Perte : entropie croisée + perte « mIoU douce » (la note officielle compte la zone marquée ET la zone propre).
Pendant l'entraînement, on note le réseau sur les images officielles de l'outil (jamais vues à l'entraînement),
avec la note officielle, et on garde le meilleur. À la fin on écrit les probabilités sur les images de test."""
import json
import time
from pathlib import Path

import numpy as np


def miou_image(p, g):
    s = 0.0
    for c in (0, 1):
        u = np.logical_or(p == c, g == c).sum()
        s += 1.0 if u == 0 else np.logical_and(p == c, g == c).sum() / u
    return s / 2


def fix_grid(prob, thr):
    """Jamais de grille vide ni pleine (aucune vraie zone ne l'est) : on garde au moins 1 % de chaque classe."""
    p = (prob >= thr).astype(np.uint8)
    k = max(1, int(0.01 * p.size))
    if p.sum() == 0:
        p.flat[np.argsort(prob.ravel())[-k:]] = 1
    elif p.sum() == p.size:
        p.flat[np.argsort(prob.ravel())[:k]] = 0
    return p


def official(root, split, method):
    from PIL import Image
    meta = [e for e in json.loads((Path(root) / split / "metadata.json").read_text()) if e["method"] == method]
    imgs = np.stack([np.asarray(Image.open(Path(root) / split / "images" / f"{e['image_name']}.png").convert("RGB")) for e in meta])
    return meta, imgs


def run(method, steps, gen_dir, official_dir, out_dir, device="cuda", bs=24, lr=3e-4, encoder="resnet50",
        max_shards=0, eval_every=500, commit=lambda: None, log=print, pretrained=True, val_limit=0, gen_sub="",
        amp_max=1.0, amp_frac=0.6, val_half=False):
    """amp_max > 1 : apprentissage progressif. Si le dataset contient les images propres, la trace du watermark est
    amplifiée (x amp_max au début, puis ramenée à x1 sur les premiers amp_frac de l'entraînement)."""
    import segmentation_models_pytorch as smp
    import torch
    import torch.nn.functional as F

    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    shards = []
    for sub in gen_sub.split(","):                           # plusieurs datasets possibles : "_off,_amp"
        found = sorted(Path(gen_dir, method + sub).glob("shard_*.npz"))
        shards += found[:max_shards] if max_shards else found
    X, Y, C = [], [], []
    for p in shards:
        z = np.load(p); X.append(z["images"]); Y.append(z["masks128"])
        if amp_max > 1 and "cleans" in z.files:
            C.append(z["cleans"])
    X = np.concatenate(X); Y = np.concatenate(Y); C = np.concatenate(C) if C else None
    log(f"{method}: {len(X)} exemples chargés en {time.time() - t0:.0f} s, part marquée {Y.mean():.3f}, "
        f"amplification {'x' + str(amp_max) if C is not None else 'non'}")

    vmeta, vimgs = official(official_dir, "train", method)
    z = np.load(Path(official_dir) / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    if val_half:                                             # seulement les images officielles jamais utilisées comme source
        par = 1 if val_half is True else int(val_half) - 1   # True ou 2 -> moitié impaire ; 1 -> moitié paire
        keep = [i for i, e in enumerate(vmeta) if int(e["image_name"], 16) % 2 == par]
        vmeta, vimgs = [vmeta[i] for i in keep], vimgs[keep]
    if val_limit:
        vmeta, vimgs = vmeta[:val_limit], vimgs[:val_limit]
    vgt = np.stack([gt[e["image_name"]] for e in vmeta])

    cuda = device.startswith("cuda")
    # 6 canaux : l'image (contexte) + ses micro-détails (image - image floutée, amplifiés x HP_GAIN), calculés en
    # pleine précision. La trace des watermarks fait 1 à 3 niveaux de gris : sans ce canal, le calcul en bf16 l'efface.
    model = smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=6, classes=1).to(device)
    if cuda:
        model = model.to(memory_format=torch.channels_last)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    ax = torch.arange(5, device=device).float() - 2
    g1 = torch.exp(-ax ** 2 / 2); g1 = g1 / g1.sum()
    blur_k = (g1[:, None] * g1[None, :]).expand(3, 1, 5, 5).contiguous()
    HP_GAIN = 20.0
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)

    def forward(xb, x01=None):
        if x01 is None:
            x01 = xb.permute(0, 3, 1, 2).float().div(255)
        hp =(x01 - F.conv2d(F.pad(x01, (2, 2, 2, 2), mode="reflect"), blur_k, groups=3)) * HP_GAIN
        x = torch.cat([(x01 - mean) / std, hp], 1)
        x = x.contiguous(memory_format=torch.channels_last) if cuda else x.contiguous()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            lg = model(x)
        return F.avg_pool2d(lg.float(), 4)[:, 0]               # 128 x 128

    @torch.no_grad()
    def predict(imgs):
        model.eval(); res = []
        for i in range(0, len(imgs), 32):
            res.append(torch.sigmoid(forward(torch.from_numpy(imgs[i:i + 32]).to(device))).cpu().numpy())
        model.train(); return np.concatenate(res)

    def evaluate(probs):
        best = (0.0, 0.5)
        for thr in np.arange(0.1, 0.91, 0.05):
            sc = float(np.mean([miou_image(fix_grid(p, thr), g) for p, g in zip(probs, vgt)]))
            best = max(best, (sc, float(thr)))
        return best

    hist, best = [], (-1.0, 0.5, 0)
    rng = np.random.default_rng(0)
    model.train(); t1 = time.time()
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(X), bs)
        xb = torch.from_numpy(X[idx]).to(device); yb = torch.from_numpy(Y[idx]).to(device).float()
        if C is not None:
            k = 1.0 + (amp_max - 1.0) * max(0.0, 1.0 - step / (amp_frac * steps))
            w = xb.permute(0, 3, 1, 2).float().div(255)
            c = torch.from_numpy(C[idx]).to(device).permute(0, 3, 1, 2).float().div(255)
            lg = forward(None, (c + k * (w - c)).clamp(0, 1))
        else:
            lg = forward(xb)
        p = torch.sigmoid(lg)
        bce = F.binary_cross_entropy_with_logits(lg, yb)
        inter1 = (p * yb).sum((1, 2)); union1 = (p + yb - p * yb).sum((1, 2))
        inter0 = ((1 - p) * (1 - yb)).sum((1, 2)); union0 = ((1 - p) + (1 - yb) - (1 - p) * (1 - yb)).sum((1, 2))
        soft_miou = 0.5 * (inter1 / (union1 + 1e-6) + inter0 / (union0 + 1e-6))
        loss = bce + (1 - soft_miou).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % eval_every == 0 or step == steps:
            sc, thr = evaluate(predict(vimgs))
            hist.append(dict(step=step, loss=float(loss), val=sc, thr=thr, s=time.time() - t1))
            log(f"{method} pas {step}: perte {float(loss):.3f} | note officielle sur leurs {len(vgt)} images {sc:.3f} "
                f"(seuil {thr:.2f}) | {(time.time() - t1) / step * 1000:.0f} ms/pas")
            if sc > best[0]:
                best = (sc, thr, step); torch.save(model.state_dict(), out / "best.pt")
            (out / "val.json").write_text(json.dumps(dict(best=best, hist=hist, n_train=len(X), encoder=encoder)))
            commit()
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    tmeta, timgs = official(official_dir, "test", method)
    if val_limit:
        tmeta, timgs = tmeta[:val_limit], timgs[:val_limit]
    np.savez_compressed(out / "test_probs.npz", names=np.array([e["image_name"] for e in tmeta]), probs=predict(timgs).astype(np.float16))
    np.savez_compressed(out / "train_probs.npz", names=np.array([e["image_name"] for e in vmeta]), probs=predict(vimgs).astype(np.float16))
    commit()
    return f"{method}: meilleure note {best[0]:.3f} (seuil {best[1]:.2f}, pas {best[2]}), {time.time() - t0:.0f} s"
