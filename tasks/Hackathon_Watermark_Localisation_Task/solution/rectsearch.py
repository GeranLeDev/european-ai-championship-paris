"""Rectangle search. Key finding (solution/croptest.py): in 'partial' images whose mask is a rectangle, the dataset
was made by crop -> embed(crop, message) -> paste back. Decoding exactly that crop (resized to the decoder's
resolution) recovers the known message with ~0.9-1.0 bit accuracy, while the full image or a far crop gives ~0.5.
So we search the box maximising the decoder's agreement with the known message:
  coarse grid of boxes (aspect <= 2.5) -> top-K -> coordinate descent on the 4 edges (steps 16, 8, 4, 2, 1).
Crops are extracted and resized in one batched call (torchvision roi_align, adaptive sampling ~ antialiasing).
Score of a box = mean(tanh(logit) * (2*bit-1)) (soft agreement in [-1, 1])."""
import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import roi_align


class Scorer:
    def __init__(self, wm, method, bs=256):
        self.wm, self.method, self.bs = wm, method, bs
        self.res = {"pixelseal": 256, "trustmark": wm.resolution if method == "trustmark" else 0, "stegastamp": 400}[method]
        self.dev = wm.device if method != "stegastamp" else ("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def net(self, crops):                                  # crops: B x 3 x R x R in [0,1] on self.dev
        if self.method == "pixelseal":
            return self.wm.det(crops)[:, 1:].float()
        if self.method == "trustmark":
            return self.wm.net(crops * 2 - 1).float()
        x = crops.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        return torch.from_numpy(self.wm._run(x, [self.wm.LOGITS])[0]).float().to(self.dev)

    @torch.no_grad()
    def score(self, x, s, boxes):
        """x: 1 x 3 x H x W float [0,1] on dev; s: nbits (+-1) on dev; boxes: N x 4 (x0, y0, x1, y1) pixel coords."""
        b = torch.as_tensor(np.asarray(boxes, np.float32), device=self.dev)
        out = []
        for i in range(0, len(b), self.bs):
            bb = b[i:i + self.bs]
            rois = torch.cat([torch.zeros(len(bb), 1, device=self.dev), bb], 1)
            crops = roi_align(x, rois, output_size=self.res, spatial_scale=1.0, sampling_ratio=-1, aligned=True)
            lg = self.net(crops)
            out.append((torch.tanh(lg) * s).mean(1))
        return torch.cat(out).cpu().numpy() if out else np.zeros(0)


def coarse_boxes(min_size=16, ratio=1.25, rel_step=0.2, max_ar=2.5, min_step=4, H=512, W=512):
    sizes = []
    v = float(min_size)
    while v < 512:
        sizes.append(int(round(v))); v *= ratio
    sizes.append(512)
    out = []
    for h in sizes:
        for w in sizes:
            if max(h / w, w / h) > max_ar: continue
            sy, sx = max(min_step, int(rel_step * h)), max(min_step, int(rel_step * w))
            ys = list(range(0, H - h + 1, sy)); xs = list(range(0, W - w + 1, sx))
            if ys[-1] != H - h: ys.append(H - h)
            if xs[-1] != W - w: xs.append(W - w)
            for y in ys:
                for x in xs:
                    out.append((x, y, x + w, y + h))
    return np.array(out, np.float32)


def refine(scorer, x, s, box, sc, steps=(16, 8, 4, 2, 1), H=512, W=512, min_size=8):
    box = np.array(box, np.float32)
    for st in steps:
        improved = True; it = 0
        while improved and it < 6:
            improved = False; it += 1
            cands = []
            for k in range(4):
                for d in (-st, st):
                    b = box.copy(); b[k] += d; cands.append(b)
            for d in (-st, st):                             # translations
                b = box.copy(); b[[0, 2]] += d; cands.append(b)
                b = box.copy(); b[[1, 3]] += d; cands.append(b)
            cands = np.array(cands)
            cands[:, [0, 2]] = cands[:, [0, 2]].clip(0, W); cands[:, [1, 3]] = cands[:, [1, 3]].clip(0, H)
            ok = ((cands[:, 2] - cands[:, 0]) >= min_size) & ((cands[:, 3] - cands[:, 1]) >= min_size)
            cands = cands[ok]
            if not len(cands): break
            v = scorer.score(x, s, cands)
            j = int(v.argmax())
            if v[j] > sc:
                sc, box, improved = float(v[j]), cands[j], True
    return box, sc


def search(scorer, img, msg, grid, topk=8, nms_iou=0.5):
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float().div(255).to(scorer.dev)
    s = torch.from_numpy(2 * np.asarray(msg, np.float32) - 1).to(scorer.dev)
    v = scorer.score(x, s, grid)
    order = np.argsort(-v)
    picked = []
    for j in order:                                         # simple NMS to diversify the starts
        if len(picked) >= topk: break
        if all(iou(grid[j], grid[p]) < nms_iou for p in picked):
            picked.append(j)
    best = (None, -9.0); coarse_best = float(v[order[0]])
    for j in picked:
        b, sc = refine(scorer, x, s, grid[j], float(v[j]))
        if sc > best[1]: best = (b, sc)
    return best[0], best[1], coarse_best


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(u, 1e-9)


def box_to_mask128(box):
    """Pixel-level rectangle -> official 128 grid (cell = 1 if >= 50 % covered)."""
    m = np.zeros((512, 512), np.float32)
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    m[max(0, y0):min(512, y1), max(0, x0):min(512, x1)] = 1
    return (m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.uint8)
