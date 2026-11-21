"""Score matrices: for every odd-half image and every post-processing config, the official mIoU.
Saved to mats/<method>.npz so that cross-validation is just column selection."""
import sys, itertools, pickle
sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/tracks/postproc")
from common import *
from pp import *
from multiprocessing import Pool

SIG = (0, 1); S = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2)); MINA = (0, 8, 32, 96); KS = (0, 1, 2, 3)
FULL_CFGS = [(sg, s, ma, k, mj, 0, 9.0) for sg, s, ma, k, mj in itertools.product(SIG, S, MINA, KS, (0, 16))]
PART_CFGS = [(sg, s, ma, k, 0, f, sn) for sg, s, ma, k, f, sn in itertools.product(SIG, S, MINA, KS, (0, 1), (9.0, 0.85, 0.7))]
DS = (-4, 0, 2, 4)


def rows_for(args):
    name, prob, g, partial, rect = args
    cfgs = PART_CFGS if partial else FULL_CFGS
    net = np.array([miou(net_mask(prob, partial, *c), g) for c in cfgs], np.float32)
    box = np.full(len(DS), np.nan, np.float32)
    if partial and rect is not None:
        for j, d in enumerate(DS):
            bg = box_grid(rect[0], d)
            if bg is not None:
                box[j] = miou(bg, g)
    return name, net, box


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    run = dict(RUN)
    for kv in sys.argv[2:]:
        k, v = kv.split("=", 1); run[k] = Path(v)
    gt, meta, acc = load_train(); rect = load_rect("train")
    Path("mats").mkdir(exist_ok=True)
    for m in METHODS:
        probs = load_probs("train", m, run)
        items = [e for e in meta if e["method"] == m and e["image_name"] in probs]
        jobs = [(e["image_name"], probs[e["image_name"]], gt[e["image_name"]], acc[e["image_name"]] < 0.8, rect.get(e["image_name"])) for e in items]
        with Pool(8) as pool:
            res = pool.map(rows_for, jobs, chunksize=2)
        names = [r[0] for r in res]; partial = np.array([j[3] for j in jobs])
        full_idx = np.where(~partial)[0]; part_idx = np.where(partial)[0]
        np.savez(f"mats/{m}{tag}.npz", names=np.array(names), partial=partial,
                 full_true=np.array([e["full_watermark"] for e in items]),
                 net_full=np.stack([res[i][1] for i in full_idx]) if len(full_idx) else np.zeros((0, len(FULL_CFGS))),
                 net_part=np.stack([res[i][1] for i in part_idx]), box=np.stack([res[i][2] for i in part_idx]),
                 rscore=np.array([rect[names[i]][1] if names[i] in rect else -1.0 for i in part_idx]),
                 full_idx=full_idx, part_idx=part_idx)
        print(m, "done", len(names), "partial(pred)", partial.sum(), flush=True)
    pickle.dump({"FULL": FULL_CFGS, "PART": PART_CFGS, "DS": DS}, open("mats/cfgs.pkl", "wb"))
