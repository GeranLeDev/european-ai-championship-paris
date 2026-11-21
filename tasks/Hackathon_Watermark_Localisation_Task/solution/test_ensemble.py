"""Tests of solution/ensemble.py (CPU, about 20 s). Run: python solution/test_ensemble.py   (or pytest solution/test_ensemble.py)
  1. the metric is the official one and equals the Modal pipeline's own implementations;
  2. the post-processing primitives equal modal_t1/tracks/postproc/pp.py;
  3. synthetic end-to-end run: clean-set intersection, cross-validation, family choice, submission format;
  4. end-to-end run on the local baseline files (skipped when the data or the caches are missing)."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import ensemble as E  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _blob(rng, frac):
    from scipy import ndimage
    x = ndimage.gaussian_filter(rng.standard_normal((128, 128)), 8)
    return (x > np.quantile(x, 1 - frac)).astype(np.uint8)


def test_metric_identical():
    from wmloc.utils import miou_image as official
    kit = _load(ROOT / "modal_t1" / "kit" / "wmloc" / "utils.py", "amil_kit_utils").miou_image       # used by combine2.py
    pp_common = _load(ROOT / "modal_t1" / "tracks" / "postproc" / "common.py", "amil_pp_common").miou  # used by the postproc track
    rng = np.random.default_rng(0)
    cases = [(rng.random((128, 128)) < rng.random(), rng.random((128, 128)) < rng.random()) for _ in range(200)]
    z, o = np.zeros((128, 128), np.uint8), np.ones((128, 128), np.uint8)
    cases += [(z, z), (o, o), (z, o), (o, z), (z, cases[0][1]), (cases[0][0], o)]
    for p, g in cases:
        ref = official(p, g)
        assert E.miou_image(p, g) == ref and abs(kit(p, g) - ref) < 1e-12 and abs(pp_common(p, g) - ref) < 1e-12
    assert official(z, o) == 0.0 and official(o, o) == 1.0


def test_postproc_matches_modal():
    pp = _load(ROOT / "modal_t1" / "tracks" / "postproc" / "pp.py", "amil_pp")
    rng = np.random.default_rng(1)
    for k in range(60):
        from scipy import ndimage
        prob = np.clip(ndimage.gaussian_filter(rng.random((128, 128)), 3) * 4 - 1.5, 0, 1).astype(np.float32)
        partial = bool(k % 2); s = float(rng.choice(E.S_GRID)); K = int(rng.integers(0, 3)); fill = int(rng.integers(0, 2))
        a = E.net_mask(prob, partial, s, K, fill); b = pp.net_mask(prob, partial, 0, s, 0, K, 0, fill)
        assert np.array_equal(a, b), (partial, s, K, fill)
        assert 0 < a.sum() < a.size
    for box, d in (((10, 20, 200, 300), 0), ((10, 20, 200, 300), 4), ((0, 0, 512, 512), 0), ((5, 5, 6, 6), 0)):
        a, b = E.box_grid(box, d), pp.box_grid(box, d)
        assert (a is None) == (b is None) and (a is None or np.array_equal(a, b))
    assert E.fix_grid(np.zeros((128, 128)), rng.random((128, 128))).sum() == 163


def _synthetic(tmp, n=60):
    """Pipeline A localises partial images well and full images badly, pipeline B the opposite; 'trustmark' only."""
    import torch
    rng = np.random.default_rng(2); data = tmp / "data"; (data / "train").mkdir(parents=True); (data / "test").mkdir()
    probs = tmp / "probs"; probs.mkdir(); glob = tmp / "glob"; glob.mkdir(); (tmp / "rect").mkdir()

    def make(names):
        full = np.array([i % 2 == 0 for i in range(len(names))])
        gts = np.stack([1 - _blob(rng, 0.08) if f else _blob(rng, 0.10) for f in full])
        noise = lambda lvl: np.stack([np.clip(g * 0.6 + 0.2 + lvl * __import__("scipy").ndimage.gaussian_filter(rng.standard_normal((128, 128)), 2) * 6, 0, 1) for g in gts])
        good, bad = noise(0.15), noise(1.2)
        pa = np.where(full[:, None, None], bad, good); pb = np.where(full[:, None, None], good, bad)
        return full, gts.astype(np.uint8), pa.astype(np.float16), pb.astype(np.float16)

    tr_names = [f"{2 * i + 1:012x}" for i in range(n)] + [f"{2 * i:012x}" for i in range(10)]         # odd, then 10 even
    te_names = [f"{1000 + i:012x}" for i in range(12)]
    full, gts, pa, pb = make(tr_names)
    np.savez_compressed(data / "train" / "masks_128.npz", names=np.array(tr_names), masks=gts)
    (data / "train" / "metadata.json").write_text(json.dumps([dict(image_name=x, method="trustmark", full_watermark=bool(f)) for x, f in zip(tr_names, full)]))
    (glob / "global_train.json").write_text(json.dumps({x: (0.97 if f else 0.55) for x, f in zip(tr_names, full)}))
    used_a = np.array([int(x, 16) % 2 == 0 for x in tr_names]); used_b = used_a.copy(); used_b[:5] = True     # B also used 5 odd images
    np.savez_compressed(probs / "a_trustmark_train.npz", names=np.array(tr_names), probs=pa, used_in_training=used_a)
    perm = rng.permutation(len(tr_names))                                                              # another order in B
    np.savez_compressed(probs / "b_trustmark_train.npz", names=np.array(tr_names)[perm], probs=pb[perm], used_in_training=used_b[perm])
    tfull, tgts, ta, tb = make(te_names)
    (data / "test" / "metadata.json").write_text(json.dumps([dict(image_name=x, method="trustmark") for x in te_names]))
    (glob / "global_test.json").write_text(json.dumps({x: (0.97 if f else 0.55) for x, f in zip(te_names, tfull)}))
    np.savez_compressed(probs / "a_trustmark_test.npz", names=np.array(te_names), probs=ta)
    np.savez_compressed(probs / "b_trustmark_test.npz", names=np.array(te_names[::-1]), probs=tb[::-1])
    torch.save({"image_names": te_names, "masks": torch.zeros(len(te_names), 128, 128, dtype=torch.bool)}, tmp / "ref.pt")
    return dict(data=data, probs=probs, glob=glob, n=n, te_names=te_names, tgts=tgts, tfull=tfull)


def test_synthetic_end_to_end():
    import torch
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); s = _synthetic(tmp)
        argv = ["eval", "--probs", str(s["probs"]), "--pipelines", "a,b", "--data", str(s["data"]), "--glob-dir", str(s["glob"]),
                "--rect-dir", str(tmp / "rect"), "--ref", str(tmp / "ref.pt"), "--test", str(tmp / "out.pt"), "--report", str(tmp / "rep.json"),
                "--repeats", "3"]
        rep = E.main(argv)["trustmark"]
        assert rep["n_clean"] == s["n"] - 5, rep["n_clean"]                       # even half and B's 5 odd images are excluded
        fam = rep["families"]
        assert fam["regime"]["cv"] > max(fam["single:a"]["cv"], fam["single:b"]["cv"]) + 0.05     # complementary by construction
        assert rep["chosen"] in ("regime", "any") and fam[rep["chosen"]]["p_gain"] >= 0.8
        assert all(f["in_sample"] >= f["cv"] - 0.02 for f in fam.values())
        assert json.loads((tmp / "rep.json").read_text())["trustmark"]["chosen"] == rep["chosen"]
        d = torch.load(tmp / "out.pt", weights_only=True)
        assert list(d.keys()) == ["image_names", "masks"] and d["image_names"] == s["te_names"]
        assert d["masks"].dtype == torch.bool and tuple(d["masks"].shape) == (12, 128, 128)
        cells = d["masks"].flatten(1).sum(1); assert int(cells.min()) > 0 and int(cells.max()) < 128 * 128
        sc = np.mean([E.miou_image(p, g) for p, g in zip(d["masks"].numpy(), s["tgts"])])
        assert sc > 0.75, sc                                                      # test maps follow the same construction
        st = E.check_submission(tmp / "out.pt", s["te_names"]); assert st["empty"] == st["full"] == 0
        for bad in (s["te_names"][::-1], s["te_names"][:-1]):                     # wrong order / wrong length must be refused
            try:
                E.check_submission(tmp / "out.pt", bad); raise SystemExit("format check did not fail")
            except AssertionError:
                pass
        # a method without any input needs --fallback; a single pipeline works and can only choose itself
        rep1 = E.main(argv[:4] + ["a"] + argv[5:-2] + ["--repeats", "2"])["trustmark"]
        assert rep1["chosen"] == "single:a" and list(rep1["families"]) == ["single:a"] and rep1["n_clean"] == s["n"]


def test_local_files_end_to_end():
    import torch
    need = [ROOT / "data" / "train" / "masks_128.npz", HERE / "cache" / "feat_train.npz", HERE / "submission_rect.pt",
            HERE / "submission_baseline.pt", ROOT / "modal_t1" / "cache" / "global_test.json"]
    if not all(p.exists() for p in need):
        print("   (skipped: local data / caches missing)"); return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        E.main(["make-demo", "--out", str(tmp / "probs")])
        z = np.load(tmp / "probs" / "rect_trustmark_train.npz")
        assert z["probs"].dtype == np.float16 and z["probs"].shape[1:] == (128, 128) and z["used_in_training"].dtype == bool
        rep = E.main(["eval", "--probs", str(tmp / "probs"), "--pipelines", "base,rect", "--test", str(tmp / "sub.pt"), "--repeats", "3"])
        assert set(rep) == set(E.METHODS)
        for m in E.METHODS:
            assert 20 <= rep[m]["n_clean"] <= 30                                  # odd half ∩ 50-image held-out fold
            assert 0.40 < rep[m]["families"][rep[m]["chosen"]]["cv"] < 0.75
        ref = torch.load(HERE / "submission_rect.pt", weights_only=True)
        st = E.check_submission(tmp / "sub.pt", ref["image_names"])
        assert st["n"] == 1020 and st["empty"] == 0 and st["full"] == 0
        # only pixelseal assembled -> the two other methods come from the fallback submission, unchanged
        for f in (tmp / "probs").glob("*_stegastamp_*.npz"): f.unlink()
        for f in (tmp / "probs").glob("*_trustmark_*.npz"): f.unlink()
        E.main(["eval", "--probs", str(tmp / "probs"), "--pipelines", "base,rect", "--test", str(tmp / "sub2.pt"), "--repeats", "2",
                "--fallback", str(HERE / "submission_rect.pt")])
        d = torch.load(tmp / "sub2.pt", weights_only=True)
        meta = json.loads((ROOT / "data" / "test" / "metadata.json").read_text())
        keep = torch.tensor([e["method"] != "pixelseal" for e in meta])
        assert torch.equal(d["masks"][keep], ref["masks"][keep]) and d["image_names"] == ref["image_names"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"--- {name}", flush=True); fn(); print(f"ok  {name}", flush=True)
    print("ALL TESTS PASSED")
