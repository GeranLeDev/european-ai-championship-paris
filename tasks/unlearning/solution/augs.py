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
