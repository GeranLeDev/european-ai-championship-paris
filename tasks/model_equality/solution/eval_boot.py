"""Bootstrap evaluation of the pipeline on a labelled answer file (simulator or ref endpoints).

Resamples the questionnaires (prompt blocks) with replacement B times and reports, for a grid of --t-model, the mean
and 10-90 % range of the task score -> pick a threshold on a plateau, not on one lucky draw.
Run: python eval_boot.py data/sim_answers.jsonl data/sim_answers_truth.csv [--gamma 2 --B 30]
     python eval_boot.py data/ref_answers.jsonl ../practice_answers.csv --prefix ref-
"""
import argparse
import random

import numpy as np

from analyze import assign, build_similarities, load_answers
from met_common import HERE, load_prompts, read_labels, task_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("truth")
    ap.add_argument("--prompts", default=str(HERE / "prompts.jsonl"))
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--B", type=int, default=30)
    ap.add_argument("--t-family", type=float, default=0.45)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--max-rep-quiz", type=int, default=2)
    ap.add_argument("--n-quiz", type=int, default=None)
    ap.add_argument("--rep-offset", type=int, default=0)
    a = ap.parse_args()
    prompts, items = load_prompts(a.prompts)
    rows = load_answers(a.answers, prompts, a.max_rep_quiz, None, a.n_quiz, a.rep_offset)
    if a.prefix:
        rows = [r for r in rows if r["endpoint"].startswith(a.prefix)]
    truth_all = read_labels(a.truth)
    eps = sorted({r["endpoint"] for r in rows})
    truth = {e: truth_all[e] for e in eps}
    qids = sorted({r["id"] for r in rows})
    by = {q: [r for r in rows if r["id"] == q] for q in qids}
    grid = np.round(np.arange(0.02, 0.42, 0.02), 2)
    scores = {t: [] for t in grid}
    fam = []
    rng = random.Random(0)
    for b in range(a.B):
        pick = [rng.choice(qids) for _ in qids] if b else qids        # draw 0 = the actual data
        rs = []
        for k, q in enumerate(pick):                                  # rename duplicated blocks
            rs += [dict(r, id=f"{q}#{k}") for r in by[q]]
        its = {f"{q}#{k}": items[q] for k, q in enumerate(pick)}
        sims, A, _ = build_similarities(rs, its, eps, a.gamma)
        S = sims["quiz_cos"]
        for t in grid:
            res = assign(eps, 1 - S, 1 - S, a.t_family, t, np.diag(A), 0.95, A)
            sc = task_score(truth, {e: (f, m) for e, f, m in res})
            scores[t].append(sc["score"])
            if t == grid[0]:
                fam.append(sc["mcc_family"])
    print(f"{len(eps)} endpoints, {len(qids)} questionnaires, B={a.B}, gamma={a.gamma}; MCC family mean {np.mean(fam):.3f}")
    print("t_model  score(actual)  mean  [p10, p90]")
    for t in grid:
        v = np.array(scores[t])
        print(f"  {t:.2f}     {v[0]:.3f}       {v.mean():.3f} [{np.percentile(v, 10):.3f}, {np.percentile(v, 90):.3f}]")


if __name__ == "__main__":
    main()
