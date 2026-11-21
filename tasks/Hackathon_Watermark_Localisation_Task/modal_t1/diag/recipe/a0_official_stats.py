import json, numpy as np
from collections import defaultdict
J = json.load(open("/Users/amil/Downloads/hackathon-watermark/task1/explore/juge_train.json"))
agg = defaultdict(list)
for r in J:
    for c in r.get("comps", []):
        y0,y1,x0,x1 = c["box"]; h,w = y1-y0, x1-x0
        agg[(r["method"], c["kind"])].append((min(h,w), max(h,w), max(h/w,w/h), c["area"]/(h*w), c["acc"], len(r["comps"]), r["acc_img"]))
for k,v in sorted(agg.items()):
    v = np.array(v)
    print(k, "n", len(v), "minside med", np.median(v[:,0]), "AR max %.1f"%v[:,2].max(), "fill med %.2f"%np.median(v[:,3]), "acc med %.3f"%np.median(v[:,4]))
    # acc vs min side
    for lo,hi in ((0,32),(32,64),(64,96),(96,128),(128,192),(192,513)):
        m=(v[:,0]>=lo)&(v[:,0]<hi)
        if m.any(): print("    minside %3d-%3d n %2d acc med %.3f  >0.9 %.2f  <0.65 %.2f | AR med %.1f"%(lo,hi,m.sum(),np.median(v[m,4]),(v[m,4]>0.9).mean(),(v[m,4]<0.65).mean(),np.median(v[m,2])))
    if k[1]=="rectangle":
        for lo,hi in ((1,1.5),(1.5,2.0),(2.0,2.05),(2.05,3),(3,5)):
            m=(v[:,2]>=lo)&(v[:,2]<hi)
            if m.any(): print("    AR %.2f-%.2f n %2d acc med %.3f >0.9 %.2f"%(lo,hi,m.sum(),np.median(v[m,4]),(v[m,4]>0.9).mean()))
# number of comps per image, kinds mix
from collections import Counter
for meth in ("trustmark","stegastamp","pixelseal"):
    c = Counter(); nc = Counter()
    for r in J:
        if r["method"]==meth and not r["full"]:
            ks = tuple(sorted(x["kind"] for x in r["comps"])); c[ks]+=1; nc[len(r["comps"])]+=1
    print(meth, dict(nc)); print("   ", c.most_common(12))
# full: acc_img vs frac
for meth in ("trustmark","stegastamp","pixelseal"):
    v = np.array([(r["frac"], r["acc_img"]) for r in J if r["method"]==meth and r["full"]])
    print(meth, "full n", len(v))
    for lo,hi in ((0.5,0.7),(0.7,0.85),(0.85,0.93),(0.93,0.97),(0.97,1.01)):
        m=(v[:,0]>=lo)&(v[:,0]<hi)
        if m.any(): print("    frac %.2f-%.2f n %2d acc med %.3f min %.3f"%(lo,hi,m.sum(),np.median(v[m,1]),v[m,1].min()))
