"""Quick diagnostic: does the per-position bit agreement of TrustMark's ResNet-50 localise the watermark, and at which input scale?"""
import sys, json, time
sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research")
from common import *
wm = load_watermark("trustmark", MODELS, DEV); n = wm.net.eval()
meta, imgs, gts, msgs = official("train", 1, "trustmark")
full = np.array([e["full_watermark"] for e in meta])
sel = list(np.where(full)[0][:12]) + list(np.where(~full)[0][:12])

@torch.no_grad()
def agree(x512, sgn, size):
    x = F.interpolate(x512, size=(size, size), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
    f = n.maxpool(n.relu(n.bn1(n.conv1(x))))
    for k in ("layer1", "layer2", "layer3", "layer4"): f = getattr(n, k)(f)
    pp = torch.einsum("bchw,oc->bohw", f, n.fc.weight) + n.fc.bias[None, :, None, None]
    gl = ((n(x) > 0).float() * 2 - 1 == sgn).float().mean()
    a = (pp * sgn[:, :, None, None]).mean(1, keepdim=True)
    return F.interpolate(a, size=(128, 128), mode="bilinear", align_corners=False)[0, 0].cpu().numpy(), float(gl)

for size in (245, 512, 768, 1024):
    res = {True: [], False: []}; t0 = time.time()
    for i in sel:
        x = to_t(imgs[i:i + 1]); sgn = torch.from_numpy(msgs[i:i + 1]).to(DEV) * 2 - 1
        a, gl = agree(x, sgn, size)
        g = gts[i]
        if 0 < g.sum() < g.size: res[bool(full[i])].append((auc(g, a), gl))
    print(f"size {size}: full AUC {np.mean([r[0] for r in res[True]]):.3f} (glob acc {np.mean([r[1] for r in res[True]]):.2f}) | "
          f"partial AUC {np.mean([r[0] for r in res[False]]):.3f} (glob {np.mean([r[1] for r in res[False]]):.2f}) [{time.time()-t0:.0f}s]", flush=True)
