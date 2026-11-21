"""Pipeline v2 (free-text probes, prompts_v2.jsonl): mechanism features -> family / model labels -> submission CSV.

Why v2: the 16-item quiz of v1 scores 0 on the labelled ref endpoints (the ref model is tiny and does not follow the
quiz format). What separates the endpoints is *how they behave*, not which facts they know.

Families
  core families = groups of endpoints that give themselves the same name in identity answers ("Who are you?" ->
  "SmolLM ... Hugging Face", "Qwen ... Alibaba Cloud"); the other endpoints (replaced system prompt, persona, heavy
  weight change) join the core family whose members answer the same prompts most like them.

Models inside a family
  ident rate r(e): share of identity answers that contain the family's identity words. Far below the family median
                   = replaced system prompt / persona = *context disguise*  -> base model.
  exact rate x(e): share of prompts whose two repeats are character-identical. >= 0.5 = greedy / beam
                   = *decoding disguise* -> base model.
  the others     : paired test over the prompts on 5 style features (log length, sentences, '!', '?', line breaks),
                   short+identity prompts and long prompts separately. z(e,f) = largest |mean difference| / s.e.
                   Same weights and same context -> z stays around 2 whatever the model (temperature and watermark
                   barely move these features on ref and in the simulator); a LoRA or a 4-bit quantisation shifts
                   the style and z grows like sqrt(number of prompts). Average linkage on z, cut at --t-z (3.0);
                   the largest cluster is the base model, every other cluster is its own model.
  Not detected (ref and simulator): fp8 quantisation and the weakest LoRA (0.3) -> merged with the base model.

  python fingerprint.py data/ref_v2.jsonl --truth ../practice_answers.csv            # score on the labelled ref
  python fingerprint.py data/ref_v2.jsonl --truth ../practice_answers.csv --curve    # score vs answers per endpoint
  python fingerprint.py data/ep_v2.jsonl --prefix ep- --out submission.csv           # real endpoints
"""
import argparse
import math
import random
import re
from collections import Counter, defaultdict

import numpy as np

from met_common import HERE, read_jsonl, read_labels, task_score, write_submission

GENERIC = {"ai", "i", "i'm", "hello", "hi", "my", "the", "as", "an", "a", "it", "this", "llm", "nlp", "you", "please"}
WORDS = re.compile(r"[a-z0-9']+")
NAME = re.compile(r"(?<=[a-z,] )[A-Z][A-Za-z]{2,}")          # capitalised word that does not start a sentence


def load(path, prompts_path, prefix=None):
    kinds = {p["id"]: p["kind"] for p in read_jsonl(prompts_path)}
    order = [p["id"] for p in read_jsonl(prompts_path)]
    rows = [r for r in read_jsonl(path) if r["id"] in kinds and not r.get("bad")
            and (prefix is None or r["endpoint"].startswith(prefix))]
    return rows, kinds, order


def group(rows):
    by = defaultdict(lambda: defaultdict(list))
    for r in sorted(rows, key=lambda r: r["repeat"]):
        by[r["endpoint"]][r["id"]].append(r["response"])
    return by


def bow(text):
    return Counter(WORDS.findall(text.lower()))


def cos(a, b):
    d = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return sum(v * b.get(k, 0) for k, v in a.items()) / d if d else 0.0


# ------------------------------------------------------------------ families
def name_rates(by, kinds, e, min_rate=0.25):
    """{name-like word: share of the endpoint's identity answers that contain it} (only words above min_rate)."""
    ans = [y for pid, ys in by[e].items() if kinds[pid] == "ident" for y in ys]
    cnt = Counter()
    for y in ans:
        cnt.update({w.lower() for w in NAME.findall(y)} - GENERIC)
    return {w: c / len(ans) for w, c in cnt.items() if ans and c / len(ans) >= min_rate}


def prompt_similarity(by, eps):
    """Mean over shared prompts of the bag-of-words cosine between answers (pairs of distinct answers), normalised
    by each endpoint's own agreement between repeats."""
    vec = {e: {pid: [bow(y) for y in ys] for pid, ys in by[e].items()} for e in eps}
    n = len(eps)
    K = np.full((n, n), np.nan)
    for i, e in enumerate(eps):
        for k in range(i, n):
            f = eps[k]
            per = []
            for pid, ye in vec[e].items():
                yf = vec[f].get(pid)
                if not yf:
                    continue
                pairs = [(a, b) for x, a in enumerate(ye) for b in ye[x + 1:]] if i == k else [(a, b) for a in ye for b in yf]
                if pairs:
                    per.append(np.mean([cos(a, b) for a, b in pairs]))
            if per:
                K[i, k] = K[k, i] = float(np.mean(per))
    d = np.sqrt(np.clip(np.nan_to_num(np.diag(K), nan=np.nanmedian(np.diag(K))), 1e-6, None))
    S = np.nan_to_num(K / d[:, None] / d[None, :])
    np.fill_diagonal(S, 1.0)
    return S


def families(by, kinds, eps, a):
    """1) Core families = groups (>= --min-core endpoints) that give themselves the same name ("Qwen ... Alibaba",
    "SmolLM ... Hugging Face") in their identity answers. 2) Every other endpoint (replaced system prompt, persona,
    heavy weight change) joins the core family whose members answer the prompts most like it.
    Fallback without any self-given name: average linkage on the prompt similarity (--t-family)."""
    S = prompt_similarity(by, eps)
    names = {e: name_rates(by, kinds, e) for e in eps}
    named = [e for e in eps if names[e]]
    fam = {}
    if len(named) >= 2:
        N = np.array([[cos(names[x], names[y]) for y in named] for x in named])
        c = cluster(1 - N, 0.5)
        k = 0
        for lab, size in Counter(c).most_common():
            if size >= a.min_core:
                for e, cl in zip(named, c):
                    if cl == lab:
                        fam[e] = k
                k += 1
    if not fam:
        c = cluster(1 - S, a.t_family)
        return {e: int(x) - 1 for e, x in zip(eps, c)}, S, {}
    core = dict(fam)
    idx = {e: i for i, e in enumerate(eps)}
    margin = {}
    for e in eps:
        if e in core:
            continue
        sc = {k: float(np.mean([S[idx[e], idx[m]] for m in core if core[m] == k])) for k in set(core.values())}
        best = max(sc, key=sc.get)
        fam[e] = best
        others = [v for k, v in sc.items() if k != best]
        margin[e] = sc[best] - max(others) if others else float("nan")
    return fam, S, margin


def cluster(D, t):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    if len(D) == 1:
        return np.array([1])
    D = np.clip((D + D.T) / 2, 0, None)
    np.fill_diagonal(D, 0.0)
    return fcluster(linkage(squareform(D, checks=False), method="average"), t=t, criterion="distance")


# ------------------------------------------------------------------ per-endpoint features inside one family
def identity_words(by, kinds, eps, min_rate=0.25):
    """Name-like words that at least half of the family's endpoints use in >= min_rate of their identity answers."""
    rate = defaultdict(list)
    for e in eps:
        ans = [y for pid, ys in by[e].items() if kinds[pid] == "ident" for y in ys]
        cnt = Counter()
        for y in ans:
            cnt.update({w.lower() for w in NAME.findall(y)} - GENERIC)
        for w, c in cnt.items():
            rate[w].append(c / max(len(ans), 1))
    need = max(2, math.ceil(len(eps) / 2)) if len(eps) > 1 else 1
    return {w for w, v in rate.items() if sum(x >= min_rate for x in v) >= need}


def features(by, kinds, eps):
    words = identity_words(by, kinds, eps)
    F = {}
    loglen = defaultdict(dict)                                   # prompt -> endpoint -> mean log length
    for e in eps:
        ident = [y for pid, ys in by[e].items() if kinds[pid] == "ident" for y in ys]
        r = np.mean([bool(words & set(WORDS.findall(y.lower()))) for y in ident]) if ident and words else float("nan")
        pairs = [ys for pid, ys in by[e].items() if kinds[pid] in ("short", "ident") and len(ys) >= 2]
        x = np.mean([ys[0] == ys[1] for ys in pairs]) if pairs else float("nan")
        sp = [ys for pid, ys in by[e].items() if kinds[pid] == "short" and len(ys) >= 2]
        h = 1 - np.mean([cos(bow(ys[0]), bow(ys[1])) for ys in sp]) if sp else float("nan")
        for pid, ys in by[e].items():
            if kinds[pid] in ("short", "ident"):
                loglen[pid][e] = np.mean([math.log(len(y) + 1) for y in ys])
        F[e] = {"ident": r, "exact": x, "noise": h}
    return F, loglen, words


STYLE = {
    "loglen": lambda y: math.log(len(y) + 1),
    "nsent": lambda y: math.log(1 + len(re.findall(r"[.!?]+", y))),
    "excl": lambda y: math.log(1 + y.count("!")),
    "quest": lambda y: math.log(1 + y.count("?")),
    "nl": lambda y: math.log(1 + y.count("\n")),
}
GROUPS = {"s": ("short", "ident"), "l": ("long",)}


def style_table(by, kinds, eps):
    """{(group, feature): {prompt: {endpoint: mean feature over the repeats}}}"""
    tab = {}
    for g, sel in GROUPS.items():
        for name, fn in STYLE.items():
            t = defaultdict(dict)
            for e in eps:
                for pid, ys in by[e].items():
                    if kinds[pid] in sel:
                        t[pid][e] = float(np.mean([fn(y) for y in ys]))
            tab[(g, name)] = t
    return tab


def pair_z(tab, e, f):
    """Paired test over the shared prompts: largest |mean difference| / standard error over the style features.
    Under 'same model, same decoding' this is the max of 10 roughly N(0,1) variables whatever the model (about 2,
    rarely above 3); a weight change that alters the style makes it grow like sqrt(number of prompts)."""
    best, arg, zs = 0.0, None, []
    for key, t in tab.items():
        d = [x[e] - x[f] for x in t.values() if e in x and f in x]
        if len(d) < 3:
            continue
        z = abs(float(np.mean(d))) / (float(np.std(d, ddof=1)) / math.sqrt(len(d)) + 0.02)
        zs.append(z)
        if z > best:
            best, arg = z, key
    if STAT == "rss":
        return math.sqrt(sum(z * z for z in zs)), arg
    if STAT == "top3":
        return float(np.mean(sorted(zs)[-3:])) if zs else 0.0, arg
    return best, arg


STAT = "max"


def assign_family(by, kinds, eps, a):
    """Returns ({endpoint: (model index, role, features)}, identity words); model 1 = base model."""
    F, _, words = features(by, kinds, eps)
    med_ident = float(np.nanmedian([F[e]["ident"] for e in eps])) if eps else float("nan")
    role = {}
    for e in eps:
        f = F[e]
        if not math.isnan(f["ident"]) and med_ident >= 0.3 and f["ident"] < a.ident_ratio * med_ident:
            role[e] = "context"          # replaced system prompt / persona -> base model
        elif not math.isnan(f["exact"]) and f["exact"] >= a.exact_min:
            role[e] = "decoding"         # greedy / beam -> base model
        else:
            role[e] = "plain"
    plain = [e for e in eps if role[e] == "plain"] or list(eps)
    tab = style_table(by, kinds, eps)
    n = len(plain)
    Z = np.zeros((n, n))
    for i in range(n):
        for k in range(i + 1, n):
            Z[i, k] = Z[k, i] = pair_z(tab, plain[i], plain[k])[0]
    c = cluster(Z, a.t_z) if n > 1 else np.array([1])
    sizes = Counter(c)
    central = {k: float(np.mean(Z[c == k][:, c != k])) if (c != k).any() else 0.0 for k in sizes}
    base = min(sizes, key=lambda k: (-sizes[k], central[k]))      # largest cluster; ties -> the most central one
    lab, nxt = {}, 2
    for k in sorted(set(c), key=lambda k: central[k]):
        if k == base:
            idx = 1
        else:
            idx, nxt = nxt, nxt + 1
        for e, kk in zip(plain, c):
            if kk == k:
                lab[e] = idx
    base_members = [e for e in plain if lab[e] == 1]
    out = {}
    for e in eps:
        zb = [pair_z(tab, e, b)[0] for b in base_members if b != e]
        zbase = float(np.mean(zb)) if zb else 0.0
        if role[e] != "plain":
            lab[e] = 1
        r = role[e] if role[e] != "plain" else ("base" if lab[e] == 1 else "changed")
        out[e] = (lab[e], r, dict(F[e], zbase=zbase))
    return out, words


def letters(i):
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def run(rows, kinds, a, report=False, kindmap=None):
    by = group(rows)
    eps = sorted(by)
    # tokenizer groups (tokprobe.py): endpoints with different tokenizers can never be one family
    sig = {}
    if getattr(a, "tok", None):
        for r in read_jsonl(a.tok):
            if r.get("exact", True):
                sig.setdefault(r["endpoint"], {})[r["unit"]] = r["accepted"]
    parts = defaultdict(list)
    for e in eps:
        parts[tuple(sorted(sig.get(e, {}).items()))].append(e)
    fam, margin, S, nfam = {}, {}, None, 0
    for key in sorted(parts):
        f_, S, m_ = families(by, kinds, parts[key], a)
        for e, k in f_.items():
            fam[e] = nfam + k
        margin.update(m_)
        nfam += max(f_.values()) + 1
        if sig:
            lines_tok = f"tokenizer group {dict(key)}: {len(parts[key])} endpoints"
            print(lines_tok) if report else None
    res, lines, pass2 = [], [], []
    for fi, f in enumerate(sorted(set(fam.values()))):
        members = [e for e in eps if fam[e] == f]
        out, words = assign_family(by, kinds, members, a)
        undecided = [e for e in members if out[e][1] == "base" and out[e][2]["zbase"] > a.t_z - 1
                     or out[e][1] == "changed" and out[e][2]["zbase"] < a.t_z + 1]
        if undecided:       # the test is paired: the base endpoints of the family must get the same extra prompts
            pass2 += [e for e in members if out[e][1] in ("base", "changed")]
        lines.append(f"family {letters(fi)} ({len(members)} endpoints) identity words: {sorted(words)}")
        for e in members:
            m, role, f_ = out[e]
            res.append((e, letters(fi), f"{letters(fi)}{m}"))
            k = f" [{kindmap[e][:22]}]" if kindmap else ""
            amb = "  <- AMBIGUOUS" if role == "base" and f_["zbase"] > a.t_z - 1 or role == "changed" and f_["zbase"] < a.t_z + 1 else ""
            att = f" (attached to family, margin {margin[e]:+.3f})" if e in margin else ""
            lines.append(f"  {e}{k:25s} -> {letters(fi)}{m} {role:8s} ident {f_['ident']:.2f} exact {f_['exact']:.2f} "
                         f"noise {f_['noise']:.2f} z-to-base {f_['zbase']:.1f}{amb}{att}")
    if report:
        print("\n".join(lines))
        print(f"pass 2 (families with an AMBIGUOUS endpoint: their base + changed endpoints, {len(pass2)} endpoints):"
              f" --endpoints {','.join(pass2) if pass2 else '(none)'}")
    return sorted(res), S


def subsample(rows, order, n, rng=None):
    """n answers per endpoint = n/2 prompts x 2 repeats. rng None -> the first n/2 prompts (the real staged plan),
    else a random set of prompts with the same ident/short/long mix as a prefix (2:5:3 per block of 10)."""
    k = n // 2
    if rng is None:
        keep = set(order[:k])
    else:
        blocks = [order[i:i + 10] for i in range(0, len(order), 10)]
        slots = [[b[j] for b in blocks if j < len(b)] for j in range(10)]     # same slot = same kind
        keep = set()
        extra = set(rng.sample(range(10), k % 10))
        for j in range(10):
            want = k // 10 + (1 if j in extra else 0)
            keep.update(rng.sample(slots[j], min(want, len(slots[j]))))
    return [r for r in rows if r["id"] in keep and r["repeat"] < 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--prompts", default=str(HERE / "prompts_v2.jsonl"))
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n", type=int, default=None, help="use only the first n answers per endpoint (n/2 prompts x 2)")
    ap.add_argument("--curve", action="store_true", help="score vs answers per endpoint (needs --truth)")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--t-family", type=float, default=0.45, help="fallback only (no self-given names)")
    ap.add_argument("--tok", default=None, help="tokprobe.py output: families are built inside tokenizer groups")
    ap.add_argument("--min-core", type=int, default=3, help="smallest group sharing a self-given name = a family")
    ap.add_argument("--t-z", type=float, default=3.0, help="paired style z above which two endpoints are two models")
    ap.add_argument("--ident-ratio", type=float, default=0.25)
    ap.add_argument("--exact-min", type=float, default=0.5)
    a = ap.parse_args()
    rows, kinds, order = load(a.answers, a.prompts, a.prefix)
    if a.n:
        rows = subsample(rows, order, a.n)
    truth = kindmap = None
    if a.truth:
        t = read_labels(a.truth)
        truth = {e: t[e] for e in {r["endpoint"] for r in rows}}
        kindmap = {l.split(",")[0]: l.strip().split(",")[3] for l in open(a.truth).readlines()[1:]}
    per = Counter(r["endpoint"] for r in rows)
    full = Counter(r["id"] for r in rows if r["repeat"] < 2)
    order = [p for p in order if full[p] == 2 * len(per)]           # prompts answered twice by every endpoint
    print(f"{len(per)} endpoints, answers/endpoint: min {min(per.values())} max {max(per.values())}")
    res, S = run(rows, kinds, a, report=True, kindmap=kindmap)
    if truth:
        sc = task_score(truth, {e: (f, m) for e, f, m in res})
        print(f"TASK SCORE {sc['score']:.3f} (MCC family {sc['mcc_family']:.3f}, MCC model {sc['mcc_model']:.3f}) "
              f"model conf tp,fp,fn,tn={sc['model_conf(tp,fp,fn,tn)']}")
        print("(a single true family gives MCC family = 0 by definition: the ref score is 0.9 x MCC model at most)")
    if a.curve and truth:
        nmax = min(per.values())
        print("\n n/endpoint  prefix   random prompt sets: mean  sd   [p10, p90]   P(score>=0.4)")
        for n in [x for x in (10, 20, 30, 40, 50, 60, 80, 100) if x <= nmax]:
            def score(rs):
                r_, _ = run(rs, kinds, a)
                return task_score(truth, {e: (f, m) for e, f, m in r_})["score"]
            pre = score(subsample(rows, order, n))
            rng = random.Random(n)
            v = np.array([score(subsample(rows, order, n, rng)) for _ in range(a.draws)]) if n < nmax else np.array([pre])
            print(f"   {n:4d}      {pre:.3f}    {v.mean():.3f}  {v.std():.3f}  [{np.percentile(v, 10):.3f}, "
                  f"{np.percentile(v, 90):.3f}]   {np.mean(v >= 0.4):.2f}")
    if a.out:
        write_submission(res, a.out)
        print("->", a.out)


if __name__ == "__main__":
    main()
