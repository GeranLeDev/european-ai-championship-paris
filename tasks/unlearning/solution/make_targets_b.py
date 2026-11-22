"""Lane B per-image target files (p2, ce) for unlearn_b.py --target-file. Run with real/ = copy of solution/cache."""
import torch
R="real/"; y=torch.load(R+"orig_forget_logits.pt").argmax(1); N=len(y); cap=torch.isin(y,torch.tensor([0,6,7,8,29,30,31,33,38]))
H={k:torch.load(R+f"gold_{k}.pt") for k in ["medium","strong","xstrong"]}
def rm(p):
    p=p.clone(); p[cap,y[cap]]=0; return p/p.sum(1,keepdim=True)
def soft(p,tau): q=p.clamp_min(1e-12)**(1/tau); return q/q.sum(1,keepdim=True)
cm,cs,cx=rm(H["medium"]["p2"]),rm(H["strong"]["p2"]),rm(H["xstrong"]["p2"])
mix=0.5*(cm+H["strong"]["p2"]); cem=H["medium"]["ce"].clone(); cem[cap]+=10; ce=torch.cat([cem,H["strong"]["ce"]],1)
cnt=torch.bincount(y,minlength=100).float(); r=(cnt/130).clamp(max=1)[y][:,None]
T={"soft15":soft(mix,1.5),"csx":0.5*(cs+cx)}
sh=(1-r)*H["medium"]["p2"]+r*H["xstrong"]["p2"]; sh[cap]=(0.5*(cm+cs))[cap]; T["share"]=sh
T["share15"]=soft(sh,1.5)
for k,q in T.items():
    assert torch.isfinite(q).all() and (q.sum(1)-1).abs().max()<1e-4
    torch.save(dict(p2=q.float(),ce=ce),f"tgt_{k}.pt"); print(k,"p_true cap %.3f part %.3f"%(q[cap,y[cap]].mean(),q[~cap,y[~cap]].mean()), "ent %.2f"%(-(q*q.clamp_min(1e-12).log()).sum(1).mean()))
print("mixcls p_true cap %.3f part %.3f"%(mix[cap,y[cap]].mean(),mix[~cap,y[~cap]].mean()), "removal share partial: min %.2f mean %.2f max %.2f"%(r[~cap].min(),r[~cap].mean(),r[~cap].max()))

# ---- partial-class forgetting targets (pw / pw2)
import torch
R="real/"; y=torch.load(R+"orig_forget_logits.pt").argmax(1); N=len(y); cap=torch.isin(y,torch.tensor([0,6,7,8,29,30,31,33,38]))
H={k:torch.load(R+f"gold_{k}.pt") for k in ["medium","strong","xstrong"]}
def rmall(p):
    p=p.clone(); p[torch.arange(N),y]=0; return p/p.sum(1,keepdim=True)
cm,cs,cx=rmall(H["medium"]["p2"]),rmall(H["strong"]["p2"]),rmall(H["xstrong"]["p2"])
cnt=torch.bincount(y,minlength=100).float(); xp=H["xstrong"]["p2"][torch.arange(N),y]+H["strong"]["p2"][torch.arange(N),y]
for name,(a0,a1) in {"pw":(0.10,0.50),"pw2":(0.20,0.70)}.items():
    q=H["medium"]["p2"].clone(); hard=torch.zeros(N,dtype=torch.bool)
    for c in y.unique().tolist():
        if cnt[c]>=129: continue
        ix=(y==c).nonzero()[:,0]; f=a0+a1*min(cnt[c].item()/130,1); k=int(round(f*len(ix))); hard[ix[xp[ix].argsort()[:k]]]=True
    wrong=0.5*(cs+cx); q[hard]=0.85*wrong[hard]+0.15*H["strong"]["p2"][hard]; q[cap]=(0.5*(cm+cs))[cap]
    q[cap]=0.5*q[cap]+0.5*H["strong"]["p2"][cap]   # capped rows = mixcls rows (public-validated shape)
    ce=torch.cat([H["medium"]["ce"],H["strong"]["ce"]],1)
    torch.save(dict(p2=q.float(),ce=ce),f"tgt_{name}.pt"); print(name,"hard share of partial %.3f"%hard[~cap].float().mean(),"p_true part hard %.3f easy %.3f cap %.3f"%(q[hard,y[hard]].mean(),q[~cap&~hard,y[~cap&~hard]].mean(),q[cap,y[cap]].mean()))
