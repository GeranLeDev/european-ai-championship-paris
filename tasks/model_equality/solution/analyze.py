"""Features -> pairwise similarities -> clustering into family / model -> submission CSV.

Signatures (all computed on the SAME fixed prompts for every endpoint):
  quiz : per item (question inside a questionnaire prompt) the canonical answer. For endpoints e,f
         A(e,f) = P(answer_e == answer_f) averaged over items (all repeat pairs), A(e,e) over distinct repeats.
         cos(e,f) = A(e,f) / sqrt(A(e,e) A(f,f))   (normalised by each endpoint's own noise -> robust to temperature)
         mmd(e,f) = A(e,e) + A(f,f) - 2 A(e,f)     (= Hamming/one-hot MMD of the paper, per item)
  free : character Hamming kernel on the first L chars of open prompts (paper's MMD-Hamming), same normalisation.
Clustering: average-linkage on d = 1 - sim; families with threshold --t-family, models (inside each family) with
--t-model.

Run:
  python analyze.py data/sim_answers.jsonl --truth data/sim_answers_truth.csv      # evaluate on the simulator
  python analyze.py data/ep_answers.jsonl --out submission.csv                       # real submission
"""
import argparse
import itertools
import json
from collections import Counter, defaultdict

import numpy as np

from met_common import HERE, endpoint_fillers, load_prompts, parse_quiz, read_jsonl, read_labels, task_score, write_submission


# ------------------------------------------------------------------ data
def load_answers(path, prompts, max_rep_quiz=None, max_rep_free=None, n_quiz=None, rep_offset=0):
    kinds = {p["id"]: p["kind"] for p in prompts}
    quiz_ids = [p["id"] for p in prompts if p["kind"] == "quiz"][:n_quiz]
    rows = []
    for r in read_jsonl(path):
        if r["repeat"] >= rep_offset:
            rows.append(dict(r, repeat=r["repeat"] - rep_offset))
    keep = []
    for r in rows:
        k = kinds.get(r["id"])
        if k == "quiz" and r["id"] in quiz_ids and (max_rep_quiz is None or r["repeat"] < max_rep_quiz):
            keep.append(r)
        elif k == "free" and (max_rep_free is None or r["repeat"] < max_rep_free):
            keep.append(r)
    return keep


def quiz_counts(rows, items):
    """{endpoint: {(qid, j): Counter(answer)}} ignoring missing answers."""
    C = defaultdict(lambda: defaultdict(Counter))
    byep = defaultdict(list)
    for r in rows:
        if r["id"] in items:
            byep[r["endpoint"]].append(r["response"])
    fill = {e: endpoint_fillers(v) for e, v in byep.items()}
    for r in rows:
        if r["id"] not in items:
            continue
        for j, a in enumerate(parse_quiz(r["response"], items[r["id"]], fill[r["endpoint"]])):
            if a not in ("<miss>", "<none>"):
                C[r["endpoint"]][(r["id"], j)][a] += 1
    return C


def agreement_matrix(C, eps, weights=None):
    """A[e,f] = mean over shared positions of P(match). Diagonal uses distinct repeats only."""
    n = len(eps)
    A = np.full((n, n), np.nan)
    for i, e in enumerate(eps):
        for k, f in enumerate(eps):
            if k < i:
                continue
            num, den = 0.0, 0.0
            for pos, ce in C[e].items():
                cf = C[f].get(pos)
                if not cf:
                    continue
                w = 1.0 if weights is None else weights.get(pos, 1.0)
                ne, nf = sum(ce.values()), sum(cf.values())
                if i == k:
                    if ne < 2:
                        continue
                    v = (sum(c * c for c in ce.values()) - ne) / (ne * (ne - 1))
                else:
                    v = sum(c * cf.get(a, 0) for a, c in ce.items()) / (ne * nf)
                num += w * v
                den += w
            A[i, k] = A[k, i] = num / den if den else np.nan
    return A


def free_matrix(rows, eps, L=200):
    """Character-level Hamming kernel (mean per position) on free prompts, same layout as agreement_matrix."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["id"].startswith("f"):
            s = r["response"][:L].ljust(L, "\0")
            by[r["endpoint"]][r["id"]].append(np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32))
    n = len(eps)
    A = np.full((n, n), np.nan)
    for i, e in enumerate(eps):
        for k, f in enumerate(eps):
            if k < i:
                continue
            vals = []
            for pid, ye in by[e].items():
                yf = by[f].get(pid)
                if not yf:
                    continue
                if i == k:
                    pairs = [(a, b) for a, b in itertools.combinations(ye, 2)]
                else:
                    pairs = [(a, b) for a in ye for b in yf]
                vals += [float(np.mean(a == b)) for a, b in pairs]
            A[i, k] = A[k, i] = np.mean(vals) if vals else np.nan
    return A


def normalise(A):
    if np.all(np.isnan(np.diag(A))):            # a single answer per prompt: no self-agreement -> raw / max
        S = A / np.nanmax(A)
        np.fill_diagonal(S, 1.0)
        return np.nan_to_num(S)
    d = np.sqrt(np.clip(np.nan_to_num(np.diag(A), nan=np.nanmedian(np.diag(A))), 1e-6, None))
    S = A / d[:, None] / d[None, :]
    np.fill_diagonal(S, 1.0)
    return np.clip(S, 0, 1.0)


def mmd_from(A):
    d = np.diag(A)
    M = d[:, None] + d[None, :] - 2 * A
    np.fill_diagonal(M, 0.0)
    return M


# ------------------------------------------------------------------ clustering
def cluster(D, t):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    if len(D) == 1:
        return np.array([1])
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=t, criterion="distance")


def letters(i):
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def assign(eps, D_fam, D_mod, t_family, t_model, selfagree=None, det_thr=0.95, A=None):
    """Families by D_fam; models inside each family by D_mod. Deterministic endpoints (self-agreement >= det_thr:
    greedy / beam decoding) are by construction decoding variants of a base model: they are not clustered with the
    noisy similarity but attached to the model group whose sampled endpoints they match best (A(g,e)/A(e,e)),
    ties -> the largest group."""
    D_mod = np.clip(D_mod, 0, None)
    fam = cluster(np.clip(D_fam, 0, None), t_family)
    rows = []
    for fi, f in enumerate(sorted(set(fam))):
        idx = np.where(fam == f)[0]
        det = [i for i in idx if selfagree is not None and selfagree[i] >= det_thr]
        rest = [i for i in idx if i not in det]
        if not rest:
            rest, det = list(idx), []
        sub = cluster(D_mod[np.ix_(rest, rest)], t_model) if len(rest) > 1 else np.array([1])
        lab = {i: int(m) for i, m in zip(rest, sub)}
        groups = defaultdict(list)
        for i, m in lab.items():
            groups[m].append(i)
        for g in det:
            def fit(m):
                mem = groups[m]
                r = np.nanmean([A[g, e] / max(A[e, e], 1e-3) for e in mem]) if A is not None else 0
                return (round(float(r), 2), len(mem))
            lab[g] = max(groups, key=fit)
        for ii in idx:
            rows.append((eps[ii], letters(fi), f"{letters(fi)}{lab[ii]}"))
    return sorted(rows)


def pair_auc(S, eps, truth):
    """AUC of similarity S for 'same model' vs 'different model' over pairs inside a true family."""
    pos, neg = [], []
    for i, j in itertools.combinations(range(len(eps)), 2):
        a, b = eps[i], eps[j]
        if truth[a][0] != truth[b][0]:
            continue
        (pos if truth[a] == truth[b] else neg).append(S[i, j])
    if not pos or not neg:
        return float("nan")
    pos, neg = np.array(pos), np.array(neg)
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def item_weights(C, eps, gamma):
    """Unsupervised weight per item = pooled self-agreement over endpoints (stable items carry the signal)."""
    if not gamma:
        return None
    acc = defaultdict(list)
    for e in eps:
        for pos, c in C[e].items():
            n = sum(c.values())
            if n >= 2:
                acc[pos].append((sum(v * v for v in c.values()) - n) / (n * (n - 1)))
    return {pos: (float(np.mean(v)) + 1e-3) ** gamma for pos, v in acc.items()}


def build_similarities(rows, items, eps, gamma=0.0):
    C = quiz_counts(rows, items)
    A = agreement_matrix(C, eps, item_weights(C, eps, gamma))
    F = free_matrix(rows, eps)
    sims = {"quiz_cos": normalise(A), "quiz_raw": A.copy(), "quiz_mmd": 1 - mmd_from(A)}
    if not np.all(np.isnan(F)):
        sims["free_cos"] = normalise(F)
        sims["free_mmd"] = 1 - mmd_from(F)
        sims["combo"] = 0.8 * sims["quiz_cos"] + 0.2 * sims["free_cos"]
    return sims, A, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--prompts", default=str(HERE / "prompts.jsonl"))
    ap.add_argument("--truth", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sim", default="quiz_cos", help="similarity used for the model level")
    ap.add_argument("--t-family", type=float, default=0.45)
    ap.add_argument("--t-model", type=float, default=0.15)
    ap.add_argument("--max-rep-quiz", type=int, default=None)
    ap.add_argument("--max-rep-free", type=int, default=None)
    ap.add_argument("--n-quiz", type=int, default=None)
    ap.add_argument("--rep-offset", type=int, default=0, help="use repeats [offset, offset+max-rep) (ref replicas)")
    ap.add_argument("--prefix", default=None, help="only endpoints with this prefix (ep-, ref-, sim-)")
    ap.add_argument("--gamma", type=float, default=0.0, help="item weight exponent (0 = uniform)")
    ap.add_argument("--det-thr", type=float, default=0.95, help="self-agreement above which an endpoint is deterministic")
    ap.add_argument("--sweep", action="store_true", help="print the score over a grid of --t-model (needs --truth)")
    ap.add_argument("--dump", default=None, help="write similarity matrices to this .json")
    a = ap.parse_args()
    prompts, items = load_prompts(a.prompts)
    rows = load_answers(a.answers, prompts, a.max_rep_quiz, a.max_rep_free, a.n_quiz, a.rep_offset)
    if a.prefix:
        rows = [r for r in rows if r["endpoint"].startswith(a.prefix)]
    eps = sorted({r["endpoint"] for r in rows})
    per = Counter(r["endpoint"] for r in rows)
    print(f"{len(eps)} endpoints, answers/endpoint: min {min(per.values())} max {max(per.values())}")
    sims, A, F = build_similarities(rows, items, eps, a.gamma)
    truth = None
    if a.truth:
        t = read_labels(a.truth)
        truth = {e: t[e] for e in eps}
        kinds = {r.split(",")[0]: r.strip().split(",")[3] for r in open(a.truth).readlines()[1:]}
        for name, S in sims.items():
            print(f"AUC same-vs-diff model (within family) {name:9s}: {pair_auc(S, eps, truth):.3f}")
        print("\nendpoint kind        self-agree(quiz)  self(free)")
        for i, e in enumerate(eps):
            print(f"  {e} {kinds[e]:10s} {A[i,i]:.3f}  {F[i,i] if not np.isnan(F[i,i]) else float('nan'):.3f}")
    S = sims[a.sim]
    D_mod = 1 - S
    D_fam = 1 - normalise(A)          # families differ on (almost) every item -> large normalised distance
    res = assign(eps, D_fam, D_mod, a.t_family, a.t_model, np.diag(A), a.det_thr, A)
    if truth:
        g = {e: (f, m) for e, f, m in res}
        sc = task_score(truth, g)
        print(f"\nTASK SCORE {sc['score']:.3f}  (MCC family {sc['mcc_family']:.3f}, MCC model {sc['mcc_model']:.3f}) "
              f"model conf tp,fp,fn,tn={sc['model_conf(tp,fp,fn,tn)']}")
        for e, f, m in res:
            print(f"  {e} {kinds[e]:10s} truth {truth[e][1]:5s} -> {m}")
    if a.sweep and truth:
        for name, Sx in sims.items():
            line = []
            for t in np.arange(0.02, 0.42, 0.02):
                g = {e: (f, m) for e, f, m in assign(eps, D_fam, 1 - Sx, a.t_family, t, np.diag(A), a.det_thr, A)}
                line.append(f"{t:.2f}:{task_score(truth, g)['score']:.2f}")
            print(f"sweep {name:9s}", " ".join(line))
    if a.dump:
        json.dump({"eps": eps, **{k: np.nan_to_num(v).round(4).tolist() for k, v in sims.items()}},
                  open(a.dump, "w"))
    if a.out:
        write_submission(res, a.out)
        print("->", a.out)


if __name__ == "__main__":
    main()
