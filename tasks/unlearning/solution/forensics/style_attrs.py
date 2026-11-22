"""Estimate the prompt 'modifier' axes (viewpoint, lighting, setting, medium) with CLIP on cached forget-set features.
Usage: PYTHONPATH=<pkgs> python style_attrs.py <dir with clip_b32_laion2b_img.npy>"""
import sys, numpy as np, torch, open_clip
from pathlib import Path
out = Path(sys.argv[1]); torch.set_num_threads(2)
F = torch.from_numpy(np.load(out / "clip_b32_laion2b_img.npy")); y = np.load(out / "pseudo.npy")
model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); tok = open_clip.get_tokenizer("ViT-B-32"); model.eval()
AX = {
 "medium": ["a photograph", "a 3D render", "a digital painting", "an oil painting", "a pencil sketch", "a cartoon illustration", "a watercolor"],
 "viewpoint": ["a close-up macro shot", "an aerial drone view from above", "a low angle shot from below", "an eye-level shot", "a wide landscape shot", "a view through a smartphone screen held in a hand"],
 "lighting": ["at golden hour sunset with backlight", "on a foggy misty morning", "at night, dark", "in bright midday daylight", "under overcast sky", "in warm indoor lamp light", "at blue hour dusk", "in the rain"],
 "setting": ["indoors on a wooden table", "on a windowsill", "in a garden", "on a stone wall or steps", "on water, a lake or the sea", "in a field or countryside", "in a studio with plain background", "in a street or courtyard", "in a forest", "in a living room", "in a cafe", "on a porch or balcony"],
 "frame": ["a photo with a black or white border frame, polaroid", "a full-bleed photo without border", "a letterboxed photo with black bars"],
}
for ax, words in AX.items():
    with torch.no_grad(): T = torch.nn.functional.normalize(model.encode_text(tok(words)), dim=-1)
    a = (100 * F @ T.T).argmax(1).numpy(); sh = np.bincount(a, minlength=len(words)) / len(a)
    print(ax, "->", "; ".join(f"{w}: {s:.2f}" for w, s in sorted(zip(words, sh), key=lambda t: -t[1])))
