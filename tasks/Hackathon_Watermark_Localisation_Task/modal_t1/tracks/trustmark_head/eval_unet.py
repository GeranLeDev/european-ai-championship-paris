import sys, json, numpy as np
sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/sprintml/starter_kit")
from wmloc.utils import miou_image
from pathlib import Path
REL = Path("/Users/amil/Downloads/hackathon-watermark/sprintml/release")
meta = [e for e in json.loads((REL/"train/metadata.json").read_text()) if e["method"]=="trustmark"]
z = np.load(REL/"train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
u = np.load(sys.argv[1]); P = dict(zip(u["names"].tolist(), u["probs"].astype(np.float32)))
odd = [e for e in meta if int(e["image_name"],16)%2==1 and e["image_name"] in P]
print(len(meta), "trustmark train;", len(odd), "odd with probs; full", np.mean([e["full_watermark"] for e in odd]))
def fix(p,t):
    g=(p>=t).astype(np.uint8); k=max(1,int(0.01*g.size))
    if g.sum()==0: g.flat[np.argsort(p.ravel())[-k:]]=1
    elif g.sum()==g.size: g.flat[np.argsort(p.ravel())[:k]]=0
    return g
for sub,name in [(odd,"all"),([e for e in odd if e["full_watermark"]],"full"),([e for e in odd if not e["full_watermark"]],"partial")]:
    best=max((np.mean([miou_image(fix(P[e["image_name"]],t),gt[e["image_name"]]) for e in sub]),t) for t in np.arange(0.05,0.96,0.05))
    print(name, len(sub), "best %.4f thr %.2f"%best)
