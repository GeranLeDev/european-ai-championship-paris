"""Smoke test of the watermark baselines on the same image + attacks.
Run from repos/videoseal: .venv/bin/python ../_smoke/baselines_smoke.py"""
import torch, time, io
import torchvision.transforms.functional as F
from PIL import Image
import videoseal
from videoseal.utils.cfg import setup_model_from_checkpoint

dev = "cuda" if torch.cuda.is_available() else "cpu"
img = Image.open("assets/imgs/1.jpg").convert("RGB").resize((512, 512))
x = F.to_tensor(img).unsqueeze(0).to(dev)

def jpeg(t, q):
    buf = io.BytesIO(); F.to_pil_image(t[0].clamp(0, 1).cpu()).save(buf, "JPEG", quality=q)
    return F.to_tensor(Image.open(buf)).unsqueeze(0).to(dev)
def crop(t, frac):  # keep `frac` of the area (center)
    h, w = t.shape[-2:]; s = frac ** 0.5
    return F.center_crop(t, [int(h * s), int(w * s)])
attacks = {
    "identity": lambda t: t,
    "jpeg40": lambda t: jpeg(t, 40),
    "crop0.32": lambda t: crop(t, 0.32),
    "bright0.5": lambda t: F.adjust_brightness(t, 0.5),
    "crop>dark>jpeg40": lambda t: jpeg(F.adjust_brightness(crop(t, 0.32), 0.5), 40),
}
for name in ["baseline/hidden", "baseline/trustmark", "baseline/wam", "baseline/mbrs", "baseline/cin", "videoseal_1.0", "pixelseal"]:
    try:
        t0 = time.time(); m = (setup_model_from_checkpoint(name) if name.startswith("baseline") else videoseal.load(name)).eval().to(dev)
        with torch.no_grad():
            out = m.embed(x, is_video=False); xw, msg = out["imgs_w"], out["msgs"]
            psnr = 10 * torch.log10(1 / ((xw.clamp(0, 1) - x) ** 2).mean()).item()
            res = []
            for a, f in attacks.items():
                pred = m.detect(f(xw.clamp(0, 1)), is_video=False)["preds"][:, 1:] > 0
                res.append(f"{a}={(pred == msg.bool()).float().mean().item():.3f}")
            clean = m.detect(x, is_video=False)["preds"][:, 1:] > 0
            res.append(f"unmarked={(clean == msg.bool()).float().mean().item():.3f}")
        print(f"{name:20s} bits={msg.shape[-1]:3d} PSNR={psnr:5.1f} " + " ".join(res) + f" ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"{name:20s} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
