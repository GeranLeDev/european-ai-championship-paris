"""Main local method: unlearning on cached features (CPU-friendly, exact w.r.t. the full model).
--level head : train final norm + head on pre-norm CLS features (cls_feats.pt)
--level mlp  : train blocks.11.{norm2, mlp, ls2} + norm + head on the CLS mid-state of the last block (mid_feats.pt)
Methods: td (target distillation toward a gold proxy at T=2 + optional CE-level term), ga (NegGrad+ with clamp), scrub.
Retain: KL(orig||sub) at T=2 on Imagenette features (+ optional L2 anchor to orig weights).
Usage: python feat_unlearn.py --level mlp --method td --target cls --alpha 10 --export ../cand_cls.pt"""
import argparse, copy, json, torch, torch.nn as nn, torch.nn.functional as F
from common import *; import scoring
torch.set_num_threads(4)


class MidTail(nn.Module):
    def __init__(s, m):
        super().__init__(); b = m.blocks[11]
        s.norm2, s.mlp, s.ls2 = copy.deepcopy(b.norm2), copy.deepcopy(b.mlp), copy.deepcopy(b.ls2)
        s.norm, s.head = copy.deepcopy(m.norm), copy.deepcopy(m.head)
    def forward(s, h):
        return s.head(s.norm(h + s.ls2(s.mlp(s.norm2(h)))))
    def export_names(s):  # tail param name -> full model name
        return {"norm2": "blocks.11.norm2", "mlp": "blocks.11.mlp", "ls2": "blocks.11.ls2", "norm": "norm", "head": "head"}


class HeadTail(nn.Module):
    def __init__(s, m):
        super().__init__(); s.norm, s.head = copy.deepcopy(m.norm), copy.deepcopy(m.head)
    def forward(s, h):
        return s.head(s.norm(h))
    def export_names(s):
        return {"norm": "norm", "head": "head"}


def build(level):
    m = load_orig("cpu"); base = (MidTail if level == "mlp" else HeadTail)(m).eval()
    C = torch.load(CACHE / ("mid_feats.pt" if level == "mlp" else "cls_feats.pt"))
    return m, base, C


def targets(name, L0, y, H):
    if name.startswith("soft:"):
        tau = float(name[5:]); return F.softmax(L0 / (2 * tau), 1), F.cross_entropy(L0 / tau, y, reduction="none")
    if name == "mix":       # hedge: average of the medium and strong proxies
        p = 0.5 * (H["medium"]["p2"] + H["strong"]["p2"]); return p, torch.cat([H["medium"]["ce"], H["strong"]["ce"]], 1).mean(1)
    if name == "mixcls":    # hedge + full removal of the 9 capped classes
        p = 0.5 * (H["cls"]["p2"] + H["strong"]["p2"]); return p, torch.cat([H["cls"]["ce"], H["strong"]["ce"]], 1).mean(1)
    return H[name]["p2"], H[name]["ce"].mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="mlp", choices=["head", "mlp"]); ap.add_argument("--method", default="td", choices=["td", "ga", "scrub"])
    ap.add_argument("--target", default="cls"); ap.add_argument("--select", default=None)
    ap.add_argument("--steps", type=int, default=800); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=10.0); ap.add_argument("--w-ce", type=float, default=0.0); ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--w-stat", type=float, default=0.0, help="td: match batch mean/std of CE to the gold-proxy mean/std")
    ap.add_argument("--cap", type=float, default=3.0); ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--eval-every", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", default=None); ap.add_argument("--json", default=None)
    a = ap.parse_args(); torch.manual_seed(a.seed)
    full, base, C = build(a.level)
    L0 = torch.load(CACHE / "orig_forget_logits.pt"); y = L0.argmax(1); N = len(y); H = scoring.make_hypotheses(CACHE)
    sel = a.select or (a.target if a.target in H else "medium")
    with torch.no_grad(): Lo = {k: base(v) for k, v in C.items()}
    q, ce_t = targets(a.target, L0, y, H)
    gce = H[sel]["ce"] if a.target in H or a.target.startswith("soft") else None
    if a.target == "mix": gce = torch.cat([H["medium"]["ce"], H["strong"]["ce"]], 1)
    if a.target == "mixcls": gce = torch.cat([H["cls"]["ce"], H["strong"]["ce"]], 1)
    g_mean, g_std = gce.mean().item(), gce.std().item()

    def score(model):
        with torch.no_grad():
            Le, La, Lr = model(C["f_clean"]), model(C["f_eval"]), model(C["r_val"])
        ret = scoring.retain_scores(Lr, Lo["r_val"]); r = {"retain": ret}
        for h, G in H.items():
            g1 = scoring.group_scores(Le, Lo["f_clean"], y, G); g2 = scoring.group_scores(La, Lo["f_eval"], y, G)
            r[h] = dict(score=scoring.full_score([g1, g2], ret), clean=g1, alt=g2)
        return r

    sub = copy.deepcopy(base).train(); p0 = [p.detach().clone() for p in sub.parameters()]
    opt = torch.optim.Adam(sub.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    views = torch.stack([C["f_clean"], C["f_aug1"], C["f_aug2"]]); R = C["r_train"]; pR = F.softmax(Lo["r_train"] / 2, 1)
    kl2 = lambda Ls, p: (p * (p.clamp_min(1e-12).log() - F.log_softmax(Ls / 2, 1))).sum(1).mean()
    best, traj = (-1, 0, None), []
    for s in range(1, a.steps + 1):
        fi = torch.randint(0, N, (a.bs,)); v = torch.randint(0, 3, (a.bs,)); hf = views[v, fi]
        ri = torch.randint(0, len(R), (a.bs,)); Lf = sub(hf); Lr = sub(R[ri])
        if a.method == "td":
            lf = kl2(Lf, q[fi])
            if a.w_ce: lf = lf + a.w_ce * ((F.cross_entropy(Lf, y[fi], reduction="none") - ce_t[fi]) ** 2).mean()
            if a.w_stat:
                cef = F.cross_entropy(Lf, y[fi], reduction="none")
                lf = lf + a.w_stat * ((cef.mean() - g_mean) ** 2 + (cef.std() - g_std) ** 2)
        elif a.method == "ga":
            lf = -F.cross_entropy(Lf, y[fi], reduction="none").clamp(max=a.cap).mean()
        else:
            with torch.no_grad(): tf = base(hf)
            lf = -F.kl_div(F.log_softmax(Lf / a.T, 1), F.softmax(tf / a.T, 1), reduction="batchmean") * a.T ** 2 * (s % 2)
        loss = lf + a.alpha * kl2(Lr, pR[ri])
        if a.l2: loss = loss + a.l2 * sum(((p - q0) ** 2).sum() for p, q0 in zip(sub.parameters(), p0))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s % a.eval_every == 0 or s == a.steps:
            r = score(sub); traj.append(dict(step=s, **{h: r[h]["score"] for h in H}, retain=r["retain"],
                                               clean_medium=r["medium"]["clean"]))
            if r[sel]["score"]["final"] > best[0]:
                best = (r[sel]["score"]["final"], s, copy.deepcopy(sub.state_dict()))
            last = r
    c = last["medium"]["clean"]
    print(f"{a.level}/{a.method}/{a.target} a={a.alpha} lr={a.lr} wce={a.w_ce} l2={a.l2} | R={last['retain']['retain']:.3f} "
          f"(agr {last['retain']['agree']:.3f} kl {last['retain']['kl']:.3f}) Facc={c['acc']:.3f} CE={c['ce_mean']:.2f}±{c['ce_std']:.2f} | "
          + " ".join(f"{h}={last[h]['score']['final']:.3f}[{last[h]['score']['ms']:.2f}/{last[h]['score']['ps']:.2f}]" for h in H)
          + f" | best[{sel}]={best[0]:.3f}@{best[1]}", flush=True)
    if a.json: json.dump(dict(args=vars(a), traj=traj), open(a.json, "w"), indent=1)
    if a.export:
        sub.load_state_dict(best[2]); sd = {k: v.clone() for k, v in full.state_dict().items()}; names = sub.export_names()
        for k, v in sub.state_dict().items():
            pre, rest = k.split(".", 1); fk = names[pre] + "." + rest
            assert fk in sd and sd[fk].shape == v.shape, fk
            sd[fk] = v.detach().float().contiguous()
        chk = tt.build_model(); chk.load_state_dict(sd, strict=True)
        assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in sd.values())
        torch.save(sd, a.export); print("exported", a.export, "(NOT uploaded)")


if __name__ == "__main__":
    main()
