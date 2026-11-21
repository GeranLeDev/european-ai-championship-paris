"""Generic text-kernel similarities between endpoints on shared prompts (any prompt kind).

K(e,f) = mean over prompts, over response pairs (distinct repeats when e == f) of k(y, y');
cos(e,f) = K(e,f)/sqrt(K(e,e)K(f,f)); mmd2 = K(e,e)+K(f,f)-2K(e,f).
Kernels: 'ham' character Hamming on the first L chars (paper), 'bow' cosine of word counts, 'tri' cosine of char 3-grams.
"""
import itertools, json, re, sys
from collections import Counter, defaultdict
import numpy as np


def feats(y, kernel, L=300):
    if kernel == "bow":
        return Counter(re.findall(r"[a-z0-9']+", y.lower()))
    if kernel == "tri":
        s = re.sub(r"\s+", " ", y.lower())
        return Counter(s[i:i + 3] for i in range(len(s) - 2))
    return y[:L].ljust(L, "\0")


def kval(a, b, kernel):
    if kernel == "ham":
        return sum(x == y for x, y in zip(a, b)) / len(a)
    num = sum(v * b.get(k, 0) for k, v in a.items())
    den = (sum(v * v for v in a.values()) * sum(v * v for v in b.values())) ** 0.5
    return num / den if den else 0.0


def kernel_matrix(rows, eps, kernel="bow", ids=None):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if ids is None or r["id"] in ids:
            by[r["endpoint"]][r["id"]].append(feats(r["response"], kernel))
    n = len(eps)
    K = np.full((n, n), np.nan)
    for i, e in enumerate(eps):
        for j in range(i, n):
            f = eps[j]
            per_prompt = []
            for pid, ye in by[e].items():
                yf = by[f].get(pid)
                if not yf:
                    continue
                pairs = list(itertools.combinations(ye, 2)) if i == j else [(a, b) for a in ye for b in yf]
                if pairs:
                    per_prompt.append(np.mean([kval(a, b, kernel) for a, b in pairs]))
            if per_prompt:
                K[i, j] = K[j, i] = np.mean(per_prompt)
    return K


def cos_from(K):
    d = np.sqrt(np.clip(np.diag(K), 1e-9, None))
    S = K / d[:, None] / d[None, :]
    np.fill_diagonal(S, 1.0)
    return S


def mmd_from(K):
    d = np.diag(K)
    M = d[:, None] + d[None, :] - 2 * K
    np.fill_diagonal(M, 0.0)
    return M


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1])]
    eps = sorted({r["endpoint"] for r in rows})
    np.set_printoptions(linewidth=220, precision=3, suppress=True)
    for kern in sys.argv[2:] or ["bow", "tri", "ham"]:
        K = kernel_matrix(rows, eps, kern)
        print(kern, "self", np.diag(K).round(3))
        print("cos\n", cos_from(K))
        print("mmd2\n", mmd_from(K))
