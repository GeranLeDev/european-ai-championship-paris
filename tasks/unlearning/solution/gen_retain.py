"""Generate in-distribution, never-seen images of the 45 forget classes with SD-Turbo (class names from forensics/class_names.json).
Use: retain proxy with orig targets (keeps unlearning instance-specific, not class-wide) and a control-set collateral probe.
Output: cache/retain_gen_u8.npy [N,224,224,3] + cache/retain_gen_meta.json (intended class per image)."""
import argparse, json, random, numpy as np, torch
from pathlib import Path
from diffusers import AutoPipelineForText2Image
ap = argparse.ArgumentParser(); ap.add_argument("--per", type=int, default=24); ap.add_argument("--device", default="mps")
ap.add_argument("--out", default="cache/retain_gen"); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
names = {int(k): v[0] for k, v in json.load(open(Path(__file__).parent / "forensics/class_names.json"))["present"].items()}
SET = ["on a wooden table", "outdoors at golden hour", "in a cozy room", "by the sea at dusk", "in a garden", "on a stone ledge",
       "in morning fog", "against a dark background", "near a window with soft light", "in a rustic interior", "at night with warm light", "on a misty lake shore"]
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16").to(a.device)
pipe.set_progress_bar_config(disable=True)
random.seed(a.seed); prompts = []
for c, n in names.items():
    n = n.split("(")[0].strip()
    for i in range(a.per):
        alt = random.choice([s.strip() for s in n.split("/")])
        prompts.append((c, f"a cinematic photo of a {alt}, {random.choice(SET)}, highly detailed, soft natural light, shallow depth of field"))
imgs, meta = [], []
for i in range(0, len(prompts), a.bs):
    chunk = prompts[i:i + a.bs]
    g = torch.Generator("cpu").manual_seed(1000 + i + 100000 * a.seed)
    out = pipe(prompt=[p for _, p in chunk], num_inference_steps=1, guidance_scale=0.0, height=512, width=512, generator=g).images
    for (c, p), im in zip(chunk, out):
        imgs.append(np.asarray(im.convert("RGB").resize((224, 224)))); meta.append([c, p])
    if i % 100 == 0: print(i, len(prompts), flush=True)
np.save(a.out + "_u8.npy", np.stack(imgs)); json.dump(meta, open(a.out + "_meta.json", "w"))
print("saved", len(imgs))
