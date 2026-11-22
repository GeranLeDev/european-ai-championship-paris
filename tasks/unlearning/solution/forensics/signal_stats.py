"""Low-level forensics on forget_images.npz: borders/letterbox, JPEG 8x8 blockiness, resampling spectrum,
exact/near duplicates, invisible-watermark (SD DWT-DCT) decode.  Usage: PYTHONPATH=<pkgs> python signal_stats.py <dir with pseudo.npy + clip feats>"""
import sys, hashlib, numpy as np
from pathlib import Path
root = Path(__file__).resolve().parents[2]; out = Path(sys.argv[1])
X = np.load(root / "forget_images.npz")["images"]; y = np.load(out / "pseudo.npy")
N = len(X)
print("shape", X.shape, "min/max", X.min(), X.max(), "mean rgb", X.reshape(-1, 3).mean(0).round(1))
# exact duplicates
h = [hashlib.md5(x.tobytes()).hexdigest() for x in X]; print("exact duplicate images:", N - len(set(h)))
# letterbox / frame borders: rows or columns at the edge that are near-constant dark or white
g = X.astype(np.float32).mean(-1)
def border(g):
    top = (g[:, :6].std(axis=(1, 2)) < 3); left = (g[:, :, :6].std(axis=(1, 2)) < 3)
    return top, left
top, left = border(g); print("constant top band >=6px: %d (%.1f%%), constant left band: %d (%.1f%%), both(frame): %d" % (top.sum(), 100 * top.mean(), left.sum(), 100 * left.mean(), (top & left).sum()))
print("  top-band mean gray of those:", np.round(g[top][:, :6].mean(axis=(1, 2))[:12]))
# JPEG blockiness: mean abs horizontal gradient at column positions mod 8
d = np.abs(np.diff(g, axis=2)).mean(axis=(0, 1)); 
print("grad by (col+1)%8:", np.round([d[np.arange(223) % 8 == k].mean() for k in range(8)], 3))
d16 = [d[np.arange(223) % 16 == k].mean() for k in range(16)]; print("grad by col%16:", np.round(d16, 3))
# radial power spectrum (high-frequency energy => resize kernel / upsampling traces)
F = np.zeros((224, 224))
for i in range(0, N, 7):
    F += np.abs(np.fft.fftshift(np.fft.fft2(g[i] - g[i].mean()))) ** 2
F /= len(range(0, N, 7)); yy, xx = np.indices(F.shape); r = np.hypot(yy - 112, xx - 112).astype(int)
rad = np.bincount(r.ravel(), F.ravel())[:113] / np.bincount(r.ravel())[:113]
print("radial log10 power at r=4,16,32,56,80,100,111:", np.round(np.log10(rad[[4, 16, 32, 56, 80, 100, 111]]), 2))
lr = np.log(np.arange(4, 100)); sl = np.polyfit(lr, np.log(rad[4:100]), 1)[0]; print("spectral slope (r 4..100): %.2f ; tail ratio P(100..111)/extrap: %.3f" % (sl, rad[100:112].mean() / np.exp(np.polyval(np.polyfit(lr, np.log(rad[4:100]), 1), np.log(105)))))
# near duplicates in CLIP space
fp = out / "clip_b32_laion2b_img.npy"
if fp.exists():
    Fc = np.load(fp); S = Fc @ Fc.T; np.fill_diagonal(S, -1)
    nn = S.max(1); same = (y[S.argmax(1)] == y)
    print("CLIP NN cosine: median %.3f, p99 %.3f, max %.3f ; NN shares pseudo-label: %.3f ; pairs>0.97: %d ; >0.95: %d" % (np.median(nn), np.quantile(nn, .99), nn.max(), same.mean(), (S > 0.97).sum() // 2, (S > 0.95).sum() // 2))
    intra = [np.mean(Fc[y == c] @ Fc[y == c].T) for c in np.unique(y)]
    print("mean intra-class CLIP cosine: capped(130) %.3f vs others %.3f" % (np.mean([v for v, c in zip(intra, np.unique(y)) if (y == c).sum() >= 129]), np.mean([v for v, c in zip(intra, np.unique(y)) if (y == c).sum() < 129])))
# invisible watermark of Stable Diffusion (dwtDct, 136 bits 'StableDiffusionV1' / 48 bits SDXL)
try:
    from imwatermark import WatermarkDecoder
    import cv2
    hits = {}
    for tag, nb in (("StableDiffusionV1", 136), ("SDV2", 32)):
        dec = WatermarkDecoder("bytes", nb); ok = 0; ber = []
        ref = np.unpackbits(np.frombuffer(tag.encode(), np.uint8))
        for i in range(0, N, 35):
            for sz in (224, 512):
                im = cv2.cvtColor(X[i], cv2.COLOR_RGB2BGR)
                if sz != 224: im = cv2.resize(im, (sz, sz), interpolation=cv2.INTER_CUBIC)
                try: b = dec.decode(im, "dwtDct")
                except Exception as e: b = b""
                bits = np.unpackbits(np.frombuffer(b, np.uint8))[:len(ref)]
                if len(bits) == len(ref): ber.append((bits != ref).mean()); ok += (b == tag.encode())
        print(f"imwatermark dwtDct '{tag}': exact hits {ok}/{len(ber)}, mean bit error {np.mean(ber):.3f} (0.5 = chance)")
    WM = 0b101100111110110010010000011110111011000110011110; ref = np.array([int(b) for b in bin(WM)[2:]])
    dec = WatermarkDecoder("bits", 48); ber = []
    for i in range(0, N, 35):
        im = cv2.resize(cv2.cvtColor(X[i], cv2.COLOR_RGB2BGR), (512, 512), interpolation=cv2.INTER_CUBIC)
        try: ber.append((np.array(dec.decode(im, "dwtDct")).astype(int) != ref).mean())
        except Exception: pass
    print(f"imwatermark SDXL 48-bit message: mean bit error {np.mean(ber):.3f}, min {np.min(ber):.3f} over {len(ber)}")
except Exception as e:
    print("imwatermark unavailable:", repr(e))
