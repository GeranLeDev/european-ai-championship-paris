"""GPU-friendly augmentations on normalized tensors [B,3,H,W] in [-1,1]."""
import math, torch, torch.nn.functional as F


def rrc(x, scale=(0.25, 1.0), ratio=(3/4, 4/3)):
    B, C, H, W = x.shape
    out = torch.empty_like(x)
    for i in range(B):
        s = torch.empty(1).uniform_(*scale).item(); r = math.exp(torch.empty(1).uniform_(math.log(ratio[0]), math.log(ratio[1])).item())
        w = min(W, int(round(math.sqrt(s * r) * W))); h = min(H, int(round(math.sqrt(s / r) * H)))
        y0 = torch.randint(0, H - h + 1, (1,)).item(); x0 = torch.randint(0, W - w + 1, (1,)).item()
        out[i] = F.interpolate(x[i:i+1, :, y0:y0+h, x0:x0+w], size=(H, W), mode="bilinear", align_corners=False)[0]
    return out


def flip(x):
    m = torch.rand(x.shape[0], device=x.device) < 0.5
    x = x.clone(); x[m] = x[m].flip(-1); return x


def color(x, s=0.4):
    B = x.shape[0]; d = x.device
    b = 1 + (torch.rand(B, 1, 1, 1, device=d) * 2 - 1) * s
    c = 1 + (torch.rand(B, 1, 1, 1, device=d) * 2 - 1) * s
    sat = 1 + (torch.rand(B, 1, 1, 1, device=d) * 2 - 1) * s
    x = (x + 1) / 2 * b
    m = x.mean(dim=(1, 2, 3), keepdim=True); x = (x - m) * c + m
    g = x.mean(1, keepdim=True); x = (x - g) * sat + g
    return x.clamp(0, 1) * 2 - 1


def blur(x, p=0.3):
    k = torch.tensor([1., 4., 6., 4., 1.], device=x.device); k = (k[:, None] * k[None]) / 256
    k = k.expand(3, 1, 5, 5)
    y = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k, groups=3)
    m = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < p).float()
    return m * y + (1 - m) * x


def noise(x, s=0.05):
    return (x + torch.randn_like(x) * s * torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 2).clamp(-1, 1)


def light(x):   # mild: what the server's "altered copies" probably look like
    return blur(color(flip(rrc(x, (0.6, 1.0))), 0.2), 0.2)


def strong(x):
    return noise(blur(color(flip(rrc(x, (0.15, 0.6))), 0.4), 0.5))


def medium(x):
    return noise(blur(color(flip(rrc(x, (0.3, 1.0))), 0.3), 0.3), 0.03)


def xstrong(x):   # stronger than 'strong' (public LB: strong >> medium, so probe further in that direction)
    return noise(blur(color(flip(rrc(x, (0.08, 0.35))), 0.5), 0.6), 0.08)


def rescale(x, lo=0.4, hi=0.9):   # down-up resize (resolution loss, JPEG-like softening)
    s = torch.empty(1).uniform_(lo, hi).item(); h = max(32, int(x.shape[-1] * s))
    return F.interpolate(F.interpolate(x, size=(h, h), mode="bilinear", align_corners=False, antialias=True), size=x.shape[-2:], mode="bicubic", align_corners=False).clamp(-1, 1)


def quant(x, q=24):   # coarse 8x8 block + colour quantisation (crude JPEG stand-in)
    y = ((x + 1) * 127.5 / q).round() * q / 127.5 - 1
    return (0.5 * y + 0.5 * F.interpolate(F.avg_pool2d(x, 2), scale_factor=2, mode="nearest")).clamp(-1, 1)


def rot(x, deg=12):   # small rotations (official hint: hidden copies use crops, flips, small rotations, colour, blur, noise)
    th = (torch.rand(x.shape[0], device=x.device) * 2 - 1) * deg * math.pi / 180; c, s = th.cos(), th.sin(); z = torch.zeros_like(c)
    g = F.affine_grid(torch.stack([torch.stack([c, -s, z], 1), torch.stack([s, c, z], 1)], 1), x.shape, align_corners=False)
    return F.grid_sample(x, g, mode='bilinear', padding_mode='reflection', align_corners=False)


def wide(x):   # broad family of 'altered copies': each sample gets one of several alteration types
    out = x.clone(); k = torch.randint(0, 7, (x.shape[0],), device=x.device)
    fns = [light, medium, lambda z: rot(flip(z)), lambda z: noise(z, 0.08), lambda z: blur(z, 1.0), lambda z: rot(light(z), 8), lambda z: color(flip(z), 0.4)]
    for i, fn in enumerate(fns):
        m = k == i
        if m.any(): out[m] = fn(x[m])
    return out


def combo(x):   # chained alterations as in the official hint (crops, flips, small rotations, colour, blur, noise) applied together
    return noise(blur(color(rot(flip(rrc(x, (0.6, 1.0))), 12), 0.3), 0.5), 0.05)


def wide2(x):   # wide + chained combo + stronger rotations
    out = x.clone(); k = torch.randint(0, 3, (x.shape[0],), device=x.device)
    for i, fn in enumerate([wide, combo, lambda z: rot(flip(z), 20)]):
        m = k == i
        if m.any(): out[m] = fn(x[m])
    return out
