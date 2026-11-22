import sys, torch
from c_lib import *
B = torch.load(FE / "soupC.pt"); print(HDR)
for n in sys.argv[1:]:
    tau = 1.0
    if "@" in n: n, tau = n.split("@"); tau = float(tau)
    D = torch.load(FE / f"{n}.pt"); print(fmt(f"{n}@{tau}", measure(D, D["W"] * tau, D["b"] * tau, B)))
