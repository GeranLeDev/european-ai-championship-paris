"""Repeated 2-fold cross-validation over the score matrices (odd half only)."""
import sys, pickle
sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/tracks/postproc")
import numpy as np
C = pickle.load(open("mats/cfgs.pkl", "rb")); FULL, PART, DS = C["FULL"], C["PART"], C["DS"]
T_GRID = [9.0] + list(np.round(np.arange(0.3, 0.96, 0.05), 2))
METHODS = ("pixelseal", "trustmark", "stegastamp")


def family(cfgs, fam):
    keep = []
    for i, (sg, s, ma, k, mj, f, sn) in enumerate(cfgs):
        ok = {"thr": sg == 0 and ma == 0 and k == 0 and mj == 0 and f == 0 and sn > 1,
              "thr+sig": ma == 0 and k == 0 and mj == 0 and f == 0 and sn > 1,
              "clean": sg == 0 and mj == 0 and f == 0 and sn > 1,
              "clean+sig": mj == 0 and f == 0 and sn > 1,
              "K": sg == 0 and ma == 0 and mj == 0 and f == 0 and sn > 1,
              "snap": sg == 0 and ma == 0 and k == 0 and mj == 0 and f == 0,
              "fill": sg == 0 and ma == 0 and k == 0 and sn > 1,
              "all": True}[fam]
        if ok: keep.append(i)
    return np.array(keep)


def part_matrix(z, fam, use_rect):
    """[n_part, n_cfg] combined score with rect override (box mode)."""
    cols = family(PART, fam); N = z["net_part"][:, cols]          # n x C
    if not use_rect:
        return N, [(PART[c], (9.0, 0, "box")) for c in cols]
    B = z["box"]; rs = z["rscore"]; mats, keys = [N], [(PART[c], (9.0, 0, "box")) for c in cols]
    for t in T_GRID[1:]:
        for j, d in enumerate(DS):
            use = (rs >= t) & ~np.isnan(B[:, j])
            M = np.where(use[:, None], np.nan_to_num(B[:, j])[:, None], N)
            mats.append(M); keys += [(PART[c], (t, d, "box")) for c in cols]
    return np.concatenate(mats, 1), keys


def cv(z, fam_full, fam_part, use_rect, seeds=20):
    n = len(z["names"]); F = z["net_full"][:, family(FULL, fam_full)]; P, _ = part_matrix(z, fam_part, use_rect)
    fi, pi = z["full_idx"], z["part_idx"]
    pos_f = {g: i for i, g in enumerate(fi)}; pos_p = {g: i for i, g in enumerate(pi)}
    out = []
    for sd in range(seeds):
        perm = np.random.default_rng(sd).permutation(n); halves = [perm[: n // 2], perm[n // 2:]]
        tot = []
        for h in (0, 1):
            tr, te = halves[h], halves[1 - h]
            for M, pos in ((F, pos_f), (P, pos_p)):
                trr = [pos[i] for i in tr if i in pos]; ter = [pos[i] for i in te if i in pos]
                if not ter: continue
                col = int(np.argmax(M[trr].mean(0))) if trr else 0
                tot += list(M[ter, col])
        out.append(np.mean(tot))
    return float(np.mean(out)), float(np.std(out))


def insample(z, fam_full, fam_part, use_rect):
    F = z["net_full"][:, family(FULL, fam_full)]; P, keys = part_matrix(z, fam_part, use_rect)
    cf = int(np.argmax(F.mean(0))); cp = int(np.argmax(P.mean(0)))
    n = len(z["names"])
    return (F[:, cf].sum() + P[:, cp].sum()) / n, FULL[family(FULL, fam_full)[cf]], keys[cp]


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    for m in METHODS:
        z = np.load(f"mats/{m}{tag}.npz")
        print(f"== {m}  (n={len(z['names'])}, pred partial {z['partial'].sum()})")
        for ff, fp, r in [("thr", "thr", False), ("thr", "thr", True), ("thr+sig", "thr+sig", True), ("K", "K", True),
                          ("clean", "clean", True), ("clean+sig", "clean+sig", True), ("thr", "snap", True),
                          ("fill", "fill", True), ("all", "all", True), ("clean", "snap", True), ("K", "thr", True), ("thr", "K", True)]:
            mu, sdv = cv(z, ff, fp, r)
            ins, cf, cp = insample(z, ff, fp, r)
            print(f"  full={ff:9s} part={fp:9s} rect={int(r)} | CV {mu:.4f} ±{sdv:.4f} | in-sample {ins:.4f} | full cfg {cf} | part cfg {cp}")
