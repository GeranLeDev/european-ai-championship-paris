import json, html, sys
from collections import defaultdict
from proj import run, S
D=json.load(open('ep.json')); fam=D['fam']; free=D['free']
T=defaultdict(lambda: defaultdict(list))
for p in free:
    for e,y in free[p].items(): T[e][p].append(html.unescape(y))
import os
for fn,pf in [('data/ep_v3.jsonl','prompts_v3.jsonl'),('data/ep_v4.jsonl','prompts_v4.jsonl')]:
    if os.path.exists(S+fn):
        for l in open(S+fn):
            r=json.loads(l); T[r['endpoint']][r['id']].append(r['response'])
cfg={'F2':('ep-28',['ep-06','ep-30','ep-07'],['ep-32','ep-22','ep-26','ep-21','ep-03','ep-28','ep-06','ep-07','ep-30']),
     'F3':('ep-05',['ep-10','ep-31','ep-27'],['ep-15','ep-13','ep-09','ep-19','ep-11','ep-05','ep-10','ep-27','ep-31']),
     'F1':(sys.argv[1] if len(sys.argv)>1 else 'ep-18',['ep-04','ep-33','ep-14'],['ep-01','ep-12','ep-17','ep-18','ep-24','ep-25','ep-04','ep-14','ep-33'])}
for F,(strong,base,cands) in cfg.items():
    run(T,strong,base,[c for c in cands if c!=strong],F+' strong='+strong+' base='+','.join(base))
