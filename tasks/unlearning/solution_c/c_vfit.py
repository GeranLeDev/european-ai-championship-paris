"""Lane C: fit of the official ms structure on public points of the soup_C family.
score = S * ms * R ; ms = 1 - (|m-gm| + |s-gs|) / (gm+gs)  (orig forget CE ~ 0±0) ; R = (agree/0.92)^a / (1+KL) * 1.057 ; ps assumed constant inside the family (absorbed in S).
Usage: python c_vfit.py [a]  -> acceptable (gm, gs) region and predictions for the queue."""
import sys, itertools, numpy as np
a = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
# name: (CE mean, CE std, agree, KL, public)
PTS = {"soup_C": (3.11, 4.94, .920, .057, .3166), "ceg_m3.8_s3.5": (3.73, 3.83, .907, .067, .3159), "pol_D_a20_s0": (4.51, 4.48, .920, .060, .3600), "pol_C_a20_s1": (5.63, 4.39, .924, .065, .3209)}
EXTRA = {k: v for k, v in (kv.split("=") for kv in sys.argv[2:])}
for k, v in EXTRA.items(): PTS[k] = tuple(float(x) for x in v.split(","))
Q = {"Ds_s1": (4.47, 4.69, .916, .06), "Da_combo_s1": (4.32, 4.96, .907, .06), "N_a20_s0": (4.89, 4.08, .898, .06), "Dn_m4.8": (4.84, 4.82, .92, .06), "Dn_m4.2": (4.19, 4.84, .92, .06), "Dn_sd4.9": (4.52, 5.15, .92, .06), "Dr_a60": (4.18, 4.94, .92, .06),
     "tau1.15(chk .3098)": (3.52, 5.66, .920, .061), "tau0.85(chk .2789)": (2.71, 4.22, .920, .105), "lam1.15": (3.64, 5.27, .898, .079)}
def ms(m, s, gm, gs): return max(0.0, 1 - (abs(m - gm) + abs(s - gs)) / (gm + gs))
def R(ag, kl): return (ag / .92) ** a * 1.057 / (1 + kl)
res = []
for gm in np.arange(3.0, 8.01, 0.05):
    for gs in np.arange(3.0, 8.01, 0.05):
        x = np.array([ms(p[0], p[1], gm, gs) * R(p[2], p[3]) for p in PTS.values()]); y = np.array([p[4] for p in PTS.values()])
        S = (x @ y) / (x @ x); rms = np.sqrt(np.mean((S * x - y) ** 2)); res.append((rms, gm, gs, S))
res.sort(); best = res[0]; print(f"a={a}  best rms {best[0]:.4f} gm {best[1]:.2f} gs {best[2]:.2f} S {best[3]:.3f} (S = ps x retain level; ceiling at perfect ms with agreement .92)")
for tol in (0.006, 0.010):
    ok = [r for r in res if r[0] <= max(tol, best[0] * 1.3)]; g = np.array(ok)
    print(f" rms<={max(tol, best[0]*1.3):.4f}: n={len(ok)} gm {g[:,1].min():.2f}-{g[:,1].max():.2f} gs {g[:,2].min():.2f}-{g[:,2].max():.2f} ceiling S {g[:,3].min():.3f}-{g[:,3].max():.3f}")
    if tol == 0.006: OK = ok
print(" fit residuals:", "  ".join(f"{k} {best[3]*ms(p[0],p[1],best[1],best[2])*R(p[2],p[3]):.3f}/{p[4]:.3f}" for k, p in PTS.items()))
print(" predictions (best [min-max over acceptable set]):")
for k, p in Q.items():
    pr = [r[3] * ms(p[0], p[1], r[1], r[2]) * R(p[2], p[3]) for r in OK]; b = best[3] * ms(p[0], p[1], best[1], best[2]) * R(p[2], p[3])
    print(f"  {k:20s} {b:.3f} [{min(pr):.3f}-{max(pr):.3f}]")
