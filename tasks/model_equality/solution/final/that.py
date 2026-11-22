import pickle, sys, json, numpy as np
fam=json.load(open('ep.json'))['fam']
TS=np.round(np.arange(0.3,1.55,0.05),2)
def that(rs):
    L=np.array([sum(r[1][k].sum() for r in rs) for k in range(len(TS))])
    k=int(np.argmax(L)); k=min(max(k,1),len(TS)-2)
    a,b,c=np.polyfit(TS[k-1:k+2],L[k-1:k+2],2); return -b/(2*a)
for F in sys.argv[1:]:
    res=pickle.load(open(f'score2_{F}.pkl','rb'))
    print('==',F,' T_hat (bootstrap se over answers) ; short answers excluded variant')
    for e in sorted(res):
        rs=res[e]; t=that(rs); rng=np.random.default_rng(0)
        bs=[that([rs[i] for i in rng.integers(0,len(rs),len(rs))]) for _ in range(200)]
        # first-token-free variant: drop first 3 tokens of each answer
        rs2=[(r[0],r[1][:,3:],r[2][:,3:]) for r in rs if r[1].shape[1]>6]
        print(f"{e} {fam[e][1][:6]:6s} {fam[e][2][:12]:12s} T={t:.3f} ({np.std(bs):.3f})   skip3: {that(rs2):.3f}  n={sum(r[1].shape[1] for r in rs)}")
