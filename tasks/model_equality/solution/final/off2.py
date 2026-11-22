import json, numpy as np, re, math
from collections import Counter, defaultdict
D=json.load(open('ep.json')); fam=D['fam']; quiz=D['quiz']
eps=sorted(fam)
A=defaultdict(dict)
for k,v in quiz.items():
    for e,lst in v['ans'].items():
        a=[x[0] for x in lst]
        A[e][k]=a if len(a)>1 else a*2      # two trials
np.set_printoptions(linewidth=250,precision=2,suppress=True)
def agree(e,f):
    num=den=0
    for k in A[e]:
        if k not in A[f]: continue
        if e==f: v=float(A[e][k][0]==A[e][k][1])
        else: v=np.mean([x==y for x in A[e][k][:2] for y in A[f][k][:2]])
        num+=v; den+=1
    return num/den
for F in ['F1','F2','F3']:
    E=[e for e in eps if fam[e][0]==F]
    M=np.array([[agree(e,f) for f in E] for e in E])
    d=np.sqrt(np.diag(M)); C=M/d[:,None]/d[None,:]
    print('==',F,'cos (x100); diag = self-agreement')
    print(' '*26+' '.join(e[-2:].rjust(4) for e in E))
    for i,e in enumerate(E):
        print(e,(fam[e][1][:6]+' '+fam[e][2][:11]).ljust(19),' '.join((f"{100*C[i,j]:4.0f}" if i!=j else f"[{100*M[i,i]:2.0f}]") for j in range(len(E))), ' mean %.0f'%(100*(C[i].sum()-1)/(len(E)-1)))
