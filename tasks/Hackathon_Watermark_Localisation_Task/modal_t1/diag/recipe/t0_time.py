import sys, time, numpy as np, torch
sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/sprintml/starter_kit")
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
torch.set_num_threads(4)
rng = np.random.default_rng(0)
img = (rng.random((512,512,3))*255).astype(np.uint8)
for name in ("stegastamp","trustmark","pixelseal"):
    t=time.time(); wm = load_watermark(name, "/Users/amil/Downloads/hackathon-watermark/sprintml/models", device="cpu"); tl=time.time()-t
    msg = rng.integers(0,2,wm.nbits)
    t=time.time()
    for _ in range(3): w = wm.embed(img, msg)
    te=(time.time()-t)/3
    t=time.time(); w2 = wm.embed(np.ascontiguousarray(img[:100,:150]), msg); te2=time.time()-t
    t=time.time()
    for _ in range(3): lg = wm.logits(to_tensor(w))
    td=(time.time()-t)/3
    x = torch.cat([to_tensor(w)]*8)
    t=time.time(); lg8 = wm.logits(x); td8=time.time()-t
    print(name, "load %.1f embed512 %.3f embed-small %.3f decode1 %.3f decode8 %.3f"%(tl,te,te2,td,td8), "acc", float(((lg[0]>0).numpy()==(msg>0)).mean()), "psnr %.1f"%(10*np.log10(255**2/np.mean((w.astype(float)-img)**2))), flush=True)
