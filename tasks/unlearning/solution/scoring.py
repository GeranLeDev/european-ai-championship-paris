"""Official task-3 formulas (from task_template.py docstring) applied to a proxy gold.
gold proxy = dict(p2=[N,100] avg softmax T=2, ce=[N,K] per-'model' CE wrt label) ; labels y = orig pseudo-labels."""
import torch, torch.nn.functional as F

FULL_CLASSES = [0, 6, 7, 8, 29, 30, 31, 33, 38]   # classes with ~130 forget images (cap) -> maybe removed entirely


def kl_rows(p, logq):  # KL(p || q) per row, p probs, logq log-probs
    return (p * (p.clamp_min(1e-12).log() - logq)).sum(1)


def group_scores(L_sub, L_orig, y, gold):
    ce_s = F.cross_entropy(L_sub, y, reduction="none"); ce_o = F.cross_entropy(L_orig, y, reduction="none")
    gm, gs = gold["ce"].mean().item(), gold["ce"].std().item()
    d = lambda ce: abs(ce.mean().item() - gm) + abs(ce.std().item() - gs)
    ms = 1 - min(max(d(ce_s) / max(d(ce_o), 1e-12), 0), 1)
    kl_s = kl_rows(gold["p2"], F.log_softmax(L_sub / 2, 1)).mean().item()
    kl_o = kl_rows(gold["p2"], F.log_softmax(L_orig / 2, 1)).mean().item()
    ps = 1 - min(max(kl_s / kl_o, 0), 1)
    return dict(ms=ms, ps=ps, ce_mean=ce_s.mean().item(), ce_std=ce_s.std().item(), acc=(L_sub.argmax(1) == y).float().mean().item(),
                kl_gold=kl_s, kl_gold_orig=kl_o)


def retain_scores(L_sub, L_orig):
    agree = (L_sub.argmax(1) == L_orig.argmax(1)).float().mean().item()
    kl = kl_rows(F.softmax(L_orig / 2, 1), F.log_softmax(L_sub / 2, 1)).mean().item()
    return dict(agree=agree, kl=kl, retain=agree * 1 / (1 + kl))


def full_score(groups, ret):
    ms = sum(g["ms"] for g in groups) / len(groups); ps = sum(g["ps"] for g in groups) / len(groups)
    return dict(forget=ms * ps, ms=ms, ps=ps, retain=ret["retain"], final=ms * ps * ret["retain"])


def make_hypotheses(cache):
    """Gold hypotheses: H_med / H_strong (orig on augmented views), H_cls (H_med + full removal of FULL_CLASSES)."""
    H = {}
    for name in ["medium", "strong"]:
        H[name] = torch.load(cache / f"gold_{name}.pt")
    y = torch.load(cache / "orig_forget_logits.pt").argmax(1)
    p2 = H["medium"]["p2"].clone(); ce = H["medium"]["ce"].clone()
    m = torch.isin(y, torch.tensor(FULL_CLASSES))
    p2[m, y[m]] = 0; p2[m] = p2[m] / p2[m].sum(1, keepdim=True)
    # class never seen -> its logit is far below: CE ~ large. Use log-ratio from renormalised mass (approx +10 nats)
    ce[m] = ce[m] + 10.0
    H["cls"] = dict(p2=p2, ce=ce)
    # public LB (21/11): mixcls 0.262 > strong 0.238 >> medium 0.080 -> evaluate the combined hypotheses too
    H["mixcls"] = dict(p2=0.5 * (H["cls"]["p2"] + H["strong"]["p2"]), ce=torch.cat([H["cls"]["ce"], H["strong"]["ce"]], 1))
    p2s = H["strong"]["p2"].clone(); ces = H["strong"]["ce"].clone()
    p2s[m, y[m]] = 0; p2s[m] = p2s[m] / p2s[m].sum(1, keepdim=True); ces[m] = ces[m] + 10.0
    H["clsstrong"] = dict(p2=p2s, ce=ces)   # strong proxy + full removal of the 9 capped classes
    return H


def remove_classes(G, y, off=10.0):
    """Gold proxy G with the capped classes fully removed (mass renormalised, CE + off nats)."""
    p2 = G["p2"].clone(); ce = G["ce"].clone(); m = torch.isin(y, torch.tensor(FULL_CLASSES))
    p2[m, y[m]] = 0; p2[m] = p2[m] / p2[m].sum(1, keepdim=True); ce[m] = ce[m] + off
    return dict(p2=p2, ce=ce)


def resolve(name, H, cache):
    """Composite hypotheses, added to H on demand:
    cls:<base>:<off>  base proxy + full removal of the 9 capped classes with a CE offset of <off> nats (cls:strong:10 == clsstrong)
    split:<capbase>:<off>:<other>  capped-class images from cls:<capbase>:<off>, all other images from <other> (may be a mix: spec)
    mix:<A>+<B>+...   average of p2, concatenation of CE columns (repeat a name to weight it; names may be cls:... specs)
    <name>            cache/gold_<name>.pt written by gold_proxy.py --names <name>"""
    if name in H:
        return H[name]
    if name.startswith("mix:"):
        Gs = [resolve(p, H, cache) for p in name[4:].split("+")]
        G = dict(p2=sum(g["p2"] for g in Gs) / len(Gs), ce=torch.cat([g["ce"] for g in Gs], 1))
    elif name.startswith("split:"):   # split:<capbase>:<off>:<otherbase> -> capped-class rows from cls:<capbase>:<off>, other rows from <otherbase>
        _, cb, off, ob = name.split(":", 3)
        A = resolve(f"cls:{cb}:{off}", H, cache); B = resolve(ob, H, cache)
        m = torch.isin(torch.load(cache / "orig_forget_logits.pt").argmax(1), torch.tensor(FULL_CLASSES))
        K = max(A["ce"].shape[1], B["ce"].shape[1]); tile = lambda c: c.repeat(1, -(-K // c.shape[1]))[:, :K]
        p2 = B["p2"].clone(); ce = tile(B["ce"]).clone(); p2[m] = A["p2"][m]; ce[m] = tile(A["ce"])[m]
        G = dict(p2=p2, ce=ce)
    elif name.startswith("cls:"):
        _, base, off = name.split(":")
        G = remove_classes(resolve(base, H, cache), torch.load(cache / "orig_forget_logits.pt").argmax(1), float(off))
    else:
        G = torch.load(cache / f"gold_{name}.pt")
    H[name] = G
    return G
