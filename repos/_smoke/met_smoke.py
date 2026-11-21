"""Smoke test Model Equality Testing: Hamming MMD + permutation p-value.
Same distribution -> high p-value; shifted distribution -> low p-value.
Run: repos/model-equality-testing/.venv/bin/python repos/_smoke/met_smoke.py
"""
import sys

import numpy as np

from model_equality_testing.algorithm import run_two_sample_test
from model_equality_testing.distribution import CompletionSample
from model_equality_testing.utils import pad_to_length, tokenize_unicode

rng = np.random.default_rng(0)
M, N, L = 5, 100, 120
WORDS = "the a time to be born die weep laugh mourn dance embrace refrain".split()


def sample(n, bias):
    """Toy 'LLM': word sampler whose distribution depends on the prompt and on `bias`."""
    prompts = rng.integers(0, M, n)
    comps = []
    for p in prompts:
        probs = np.ones(len(WORDS))
        probs[p] += 3
        probs[-1] += bias
        probs /= probs.sum()
        comps.append(" ".join(rng.choice(WORDS, 10, p=probs)))
    c = pad_to_length(tokenize_unicode(comps), L=L)
    return CompletionSample(prompts=prompts, completions=c, m=M)


def test(a, b):
    p, stat = run_two_sample_test(a, b, pvalue_type="permutation_pvalue", stat_type="mmd_hamming", b=200)
    return p, stat


p_same, s_same = test(sample(N, 0), sample(N, 0))
p_diff, s_diff = test(sample(N, 0), sample(N, 4))
print(f"P vs P       : p={p_same:.3f}  MMD={s_same:.4f}")
print(f"P vs shifted : p={p_diff:.3f}  MMD={s_diff:.4f}")
ok = p_same > 0.05 and p_diff < 0.05
print("SMOKE", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
