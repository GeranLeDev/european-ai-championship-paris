"""Lane C: V model with altered copies. ms = mean over (clean, altered) of 1 - (|m-gm|+|s-gs|)/(gm+gs); altered stats = clean - k * (measured combo gap); k in [0,1] fitted.
score = S * ms * 1.057/(1+KL). Usage: python c_vfit2.py"""
import numpy as np
# name: (m, s, combo-altered m, s (None -> 0.83*m, 0.82*s), KL gen, public)
PTS = {"soup_C": (3.11, 4.94, 2.65, 3.90, .057, .3166), "ceg_m3.8_s3.5": (3.73, 3.83, None, None, .067, .3159), "pol_D": (4.51, 4.48, None, None, .060, .3600), "pol_C": (5.63, 4.39, None, None, .065, .3209),
       "pol_Ds_s1": (4.47, 4.69, 3.58, 4.00, .060, .3602), "tau1.15": (3.52, 5.66, 2.97, 4.47, .061, .3098), "tau0.85": (2.71, 4.22, 2.35, 3.32, .105, .2789),
       "Da_combo_s1": (4.32, 4.96, 3.85, 4.20, .062, .3581)}   # agreement 0.907 (others 0.916-0.924): not modelled
Q = {"Da_combo_s1": (4.32, 4.96, 3.85, 4.20, .062), "N_a20_s0": (4.89, 4.08, None, None, .065), "Dn_m4.8": (4.84, 4.82, None, None, .06), "Dn_m4.2": (4.19, 4.84, None, None, .06), "Dn_sd4.9": (4.52, 5.15, None, None, .06), "Dr_a60": (4.18, 4.94, None, None, .06),
     "ideal clean=alt=gold": None}
def alt(p): return (p[2], p[3]) if p[2] is not None else (0.83 * p[0], 0.82 * p[1])
def ms(p, gm, gs, k):
    am, as_ = alt(p); am = p[0] - k * (p[0] - am); as_ = p[1] - k * (p[1] - as_); D = gm + gs
    return max(0, 1 - 0.5 * (abs(p[0] - gm) + abs(p[1] - gs) + abs(am - gm) + abs(as_ - gs)) / D)
def pred(p, r): return r[4] * ms(p, r[1], r[2], r[3]) * 1.057 / (1 + p[4])
y = np.array([p[5] for p in PTS.values()]); res = []
for k in np.arange(0, 1.01, 0.1):
    for gm in np.arange(3.5, 5.51, 0.05):
        for gs in np.arange(3.5, 6.01, 0.05):
            x = np.array([ms(p, gm, gs, k) * 1.057 / (1 + p[4]) for p in PTS.values()]); S = (x @ y) / (x @ x); res.append((np.sqrt(np.mean((S * x - y) ** 2)), gm, gs, k, S))
res.sort(); b = res[0]; print(f"best rms {b[0]:.4f} gm {b[1]:.2f} gs {b[2]:.2f} k {b[3]:.1f} S {b[4]:.3f}")
for k in np.arange(0, 1.01, 0.2):
    r = min(x for x in res if abs(x[3] - k) < 1e-6); print(f"  k={k:.1f}: rms {r[0]:.4f} gm {r[1]:.2f} gs {r[2]:.2f} ceiling S {r[4]:.3f} | Dn_m4.8 {pred(Q['Dn_m4.8'], r):.3f} Dn_m4.2 {pred(Q['Dn_m4.2'], r):.3f} Da_combo {pred(Q['Da_combo_s1'], r):.3f} Dn_sd4.9 {pred(Q['Dn_sd4.9'], r):.3f} N {pred(Q['N_a20_s0'], r):.3f} | pol_D {pred(PTS['pol_D'], r):.3f} Ds {pred(PTS['pol_Ds_s1'], r):.3f}")
OK = [r for r in res if r[0] <= b[0] + 0.0015]; g = np.array(OK); print(f"acceptable (rms<=best+0.0015, n={len(OK)}): gm {g[:,1].min():.2f}-{g[:,1].max():.2f} gs {g[:,2].min():.2f}-{g[:,2].max():.2f} k {g[:,3].min():.1f}-{g[:,3].max():.1f} S {g[:,4].min():.3f}-{g[:,4].max():.3f}")
print("residuals:", "  ".join(f"{n} {pred(p, b):.3f}/{p[5]:.4f}" for n, p in PTS.items()))
for n, p in Q.items():
    if p is None: continue
    pr = [pred(p, r) for r in OK]; print(f"  {n:14s} {pred(p, b):.3f} [{min(pr):.3f}-{max(pr):.3f}]")
