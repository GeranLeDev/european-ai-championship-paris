"""Matched filter: re-embed the known message into the observed image to get the TrustMark residual r_hat
(deterministic encoder, cover ~ clean), then local correlation of the observed image's detail with r_hat."""
import sys, time
sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research")
from common import *
from torchvision import transforms
wm = load_watermark("trustmark", MODELS, DEV); tm = wm.tm
print("has removal:", hasattr(tm, "removal"), getattr(tm, "model_resolution_remove", None), tm.model_resolution_enc)
meta, imgs, gts, msgs = official("train", 1, "trustmark")
full = np.array([e["full_watermark"] for e in meta])

@torch.no_grad()
def residual(img_u8, msg):
    pil = Image.fromarray(img_u8)
    cover = pil.resize((tm.model_resolution_enc,) * 2, Image.BILINEAR)
    cover = transforms.ToTensor()(cover).unsqueeze(0).to(tm.encoder.device) * 2 - 1
    stego, _ = tm.encoder(cover, torch.from_numpy(msg).float().to(tm.encoder.device)[None])
    r = stego.clamp(-1, 1) - cover; r = r - r.mean((2, 3), keepdim=True)
    return F.interpolate(r, size=(512, 512), mode="bilinear")[0] * 127.5   # 3 x 512 x 512 gray levels

def blur(x, s):
    k = int(3 * s) * 2 + 1; ax = torch.arange(k, device=x.device).float() - k // 2
    g = torch.exp(-ax ** 2 / (2 * s * s)); g = g / g.sum()
    c = x.shape[0]; x = x[None]
    x = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).expand(c, 1, 1, k), groups=c)
    x = F.conv2d(x, g.view(1, 1, k, 1).expand(c, 1, k, 1), groups=c)
    return x[0]

res = {True: [], False: []}
for i in list(np.where(full)[0][:16]) + list(np.where(~full)[0][:10]):
    y = torch.from_numpy(imgs[i]).float().permute(2, 0, 1).to(DEV)
    r = residual(imgs[i], msgs[i])
    out = []
    for hs in (1.0, 2.0):
        yh = y - blur(y, hs); rh = r - blur(r, hs)
        for ws in (4.0, 8.0):
            num = blur((yh * rh).sum(0, keepdim=True), ws); den = blur((rh * rh).sum(0, keepdim=True), ws) + 1e-3
            m = (num / den)[0]
            m128 = F.avg_pool2d(m[None, None], 4)[0, 0].cpu().numpy()
            g = gts[i]
            out.append(auc(g, m128) if 0 < g.sum() < g.size else np.nan)
    res[bool(full[i])].append(out)
    print(i, "full" if full[i] else "part", np.round(out, 3), "r std %.2f" % float(r.std()), flush=True)
for k in (True, False):
    print("full" if k else "partial", "mean AUC (hp,ws)=(1,4),(1,8),(2,4),(2,8):", np.round(np.nanmean(res[k], 0), 3))
