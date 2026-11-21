"""Sur les images officielles 'completes' (toute l'image marquee sauf des trous) : sans aucun entrainement,
la trace recalculee avec le message connu permet-elle de retrouver les trous ?
Cartes testees (grille 128) :
  beta  : a quel point l'image contient la trace recalculee, par fenetre (filtre adapte)
  ppl   : accord entre le message et les sorties 'par position' du decodeur (indice donne par le kit)
Usage : python empreinte_officielle.py <starter_kit> <release> <models> [nb par methode]"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

KIT, DATA, MODELS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 30
sys.path.insert(0, str(KIT))
from wmloc.data import LocalisationDataset
from wmloc.utils import miou_image, to_tensor
from wmloc.watermarks import load_watermark

torch.set_num_threads(4)

def gauss(x, sigma):
    k = int(4 * sigma) | 1; a = torch.arange(k) - k // 2
    g = torch.exp(-a.float() ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)

def beta_map(img, trace, win, sigma=2.0):
    """Coefficient de regression local de l'image (passe-haut) sur la trace (passe-haut), sur la grille 128."""
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float(); q = torch.from_numpy(trace).permute(2, 0, 1)[None].float()
    h = x - gauss(x, sigma); q = q - gauss(q, sigma)
    box = lambda z: F.avg_pool2d(z.sum(1, keepdim=True), win, stride=4, padding=(win - 4) // 2, count_include_pad=False)
    return (box(h * q) / (box(q * q) + 1e-3))[0, 0].numpy()

def auc(score, gt):
    s = score.ravel(); g = gt.ravel().astype(bool)
    if g.all() or not g.any(): return np.nan
    r = np.argsort(np.argsort(s)) + 1
    return float((r[g].sum() - g.sum() * (g.sum() + 1) / 2) / (g.sum() * (~g).sum()))

def best_miou(maps, gts):
    ts = np.quantile(np.concatenate([m.ravel() for m in maps]), np.linspace(0.02, 0.98, 49))
    sc = [np.mean([miou_image(m > t, g) for m, g in zip(maps, gts)]) for t in ts]
    return float(np.max(sc))

for name in ("pixelseal", "trustmark", "stegastamp"):
    ds = LocalisationDataset(DATA, "train", methods=[name]); wm = load_watermark(name, MODELS, device="cpu")
    items = [ds[i] for i in range(len(ds)) if ds.meta[i]["full_watermark"]][:NB]
    res = {"beta16": [], "beta32": [], "ppl": []}; gts = []
    for it in items:
        img, msg = it["image"], it["message"]
        trace = wm.embed(img, msg).astype(np.float32) - img
        res["beta16"].append(beta_map(img.astype(np.float32), trace, 16)); res["beta32"].append(beta_map(img.astype(np.float32), trace, 32))
        if name != "stegastamp":
            ppl = wm.features(to_tensor(img))["per_position_logits"][0].cpu()
            s = (ppl * torch.from_numpy(msg.astype(np.float32) * 2 - 1)[:, None, None]).mean(0)
            res["ppl"].append(F.interpolate(s[None, None], size=(128, 128), mode="bilinear", align_corners=False)[0, 0].numpy())
            ppl_shape = tuple(ppl.shape)
        gts.append(it["mask128"])
    print(f"== {name} ({len(items)} images completes){'' if name == 'stegastamp' else f' | sorties par position : {ppl_shape}'}")
    print(f"   reference 'tout a 1' : mIoU {np.mean([miou_image(np.ones_like(g), g) for g in gts]):.3f}")
    for k, maps in res.items():
        if not maps: continue
        print(f"   {k:7s}: AUC mediane {np.nanmedian([auc(m, g) for m, g in zip(maps, gts)]):.3f} | meilleur mIoU avec un seul seuil {best_miou(maps, gts):.3f}")
