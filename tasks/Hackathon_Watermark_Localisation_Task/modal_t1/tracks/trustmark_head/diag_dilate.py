import sys, time
sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/pixelseal_research"); sys.path.insert(0, ".")
from common import *
from tm_common import dilate_resnet, resnet_feats
wm = load_watermark("trustmark", MODELS, DEV); n32 = wm.net.eval()
meta, imgs, gts, msgs = official("train", 1, "trustmark")
full = np.array([e["full_watermark"] for e in meta])
sel = list(np.where(full)[0][:16]) + list(np.where(~full)[0][:16])
nets = {os: dilate_resnet(n32, os).eval() for os in (8, 16)}; nets[32] = n32
with torch.no_grad():
    x = to_t(imgs[:2]); x = F.interpolate(x, size=(245, 245), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
    f32 = resnet_feats(n32, x)[-1]; f8 = resnet_feats(nets[8], x)[-1]
    print("shapes", f32.shape, f8.shape, "logit check", float((n32.fc(f32.mean((2,3))) - n32.fc(f8[:, :, ::4, ::4].mean((2,3)))).abs().max()))
for os, n in nets.items():
    res = {True: [], False: []}
    for i in sel:
        with torch.no_grad():
            x = F.interpolate(to_t(imgs[i:i+1]), size=(245, 245), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
            f = resnet_feats(n, x)[-1]; sgn = torch.from_numpy(msgs[i:i+1]).to(DEV) * 2 - 1
            pp = torch.einsum("bchw,oc->bohw", f, n.fc.weight) + n.fc.bias[None, :, None, None]
            a = F.interpolate((pp * sgn[:, :, None, None]).mean(1, keepdim=True), size=(128, 128), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        g = gts[i]
        if 0 < g.sum() < g.size: res[bool(full[i])].append(auc(g, a))
    print(f"OS{os}: grid {f.shape[-1]} full AUC {np.mean(res[True]):.3f} partial AUC {np.mean(res[False]):.3f}", flush=True)
