import sys
sys.path.insert(0, "."); sys.argv = ["x"]
exec(open("diag_mf.py").read().split("res = {True")[0])
import glob
fs = sorted(glob.glob("/Users/amil/Downloads/val2017/*.jpg"))[:4]
for f in fs:
    c = np.asarray(Image.open(f).convert("RGB").resize((512, 512)))
    msg = np.random.default_rng(0).integers(0, 2, 100)
    y = wm.embed(c, msg)
    r = torch.from_numpy(y.astype(np.float32) - c.astype(np.float32)).permute(2, 0, 1).to(DEV)
    r1 = residual(c, msg.astype(np.float32)); r2 = residual(y, msg.astype(np.float32))
    msg2 = 1 - msg; r3 = residual(y, msg2.astype(np.float32))
    cc = lambda a, b: float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))
    print(f"corr(true r, r from clean) {cc(r, r1):.3f} | corr(true r, r re-embedded on stego) {cc(r, r2):.3f} | other msg {cc(r, r3):.3f}")
