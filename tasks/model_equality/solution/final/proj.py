import json, re, sys, math, numpy as np
from collections import Counter, defaultdict
S='/Users/matisgeran/Pro/Hackathon/European_AI_Championship_Paris/tasks/model_equality/solution/'
W=lambda s: re.findall(r"[a-z']+|[.,!?;:\n]", s.lower())
def run(texts, strong, base, cands, label, nboot=200, seed=0):
    # texts: {endpoint: {prompt: [answers]}}; paired by prompt
    prompts=sorted(set.intersection(*[set(texts[e]) for e in [strong]+base+cands]))
    vocab=Counter()
    for e in [strong]+base:
        for p in prompts:
            for y in texts[e][p]: vocab.update(set(W(y)))
    V=[w for w,c in vocab.items() if c>=8]; idx={w:i for i,w in enumerate(V)}
    def vec(e,ps):
        c=np.zeros(len(V)); n=0
        for p in ps:
            for y in texts[e][p]:
                ws=W(y); n+=len(ws)
                for w in ws:
                    if w in idx: c[idx[w]]+=1
        return np.log((c+0.5)/(n+0.5*len(V)))
    def proj(ps):
        b=np.mean([vec(e,ps) for e in base],0); d=vec(strong,ps)-b
        return {e:float((vec(e,ps)-np.mean([vec(f,ps) for f in base if f!=e],0))@d/(d@d)) for e in cands}
    full=proj(prompts); rng=np.random.default_rng(seed); bs=defaultdict(list)
    for _ in range(nboot):
        ps=[prompts[i] for i in rng.integers(0,len(prompts),len(prompts))]
        for e,v in proj(ps).items(): bs[e].append(v)
    print('==',label,'| projection on the direction (strong LoRA - base group), 1 = as far as the strong LoRA; n prompts',len(prompts),'vocab',len(V))
    for e in sorted(cands,key=lambda e:full[e]): print(f"   {e} {full[e]:+.3f} (se {np.std(bs[e]):.3f})")
if __name__=='__main__':
    rows=[json.loads(l) for l in open(S+'data/ref_v2.jsonl')]
    T=defaultdict(lambda: defaultdict(list))
    for r in rows: T[r['endpoint']][r['id']].append(r['response'])
    kind='01 plain,02 fp8,03 sys,04 L0.7,05 pers,06 WM,07 L1.5,08 T1,09 L0.3'
    print(kind)
    run(T,'ref-07',['ref-01','ref-02','ref-06','ref-08'],['ref-01','ref-02','ref-06','ref-08','ref-09','ref-04','ref-03','ref-05'],'ref, base=01,02,06,08')
    run(T,'ref-07',['ref-06','ref-08','ref-03','ref-05'],['ref-01','ref-02','ref-09','ref-04'],'ref, base = disguises only (06,08,03,05) as on ep')
