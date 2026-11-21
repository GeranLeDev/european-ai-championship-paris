"""Summarise JURECA runs: for each output/runs/<tag>_g<i>/hist.json print config + best proxy score under each gold hypothesis.
Usage (from unlearning/): python solution/collect.py"""
import json, glob, os
rows = []
for f in sorted(glob.glob("output/runs/*/hist.json")):
    d = json.load(open(f)); a = d["args"]; best = d["best"]; h = {x["step"]: x for x in d["hist"]}[best[1]] if best[1] in {x["step"] for x in d["hist"]} else d["hist"][-1]
    cfg = f"tgt={a['target']} wstat={a.get('w_stat',0)} a={a['alpha']} lr={a['lr']} steps={a['steps']} seed={a['seed']}"
    sc = " ".join(f"{k}={h[k]['score']['final']:.3f}" for k in ["medium", "strong", "cls"])
    rows.append((os.path.dirname(f), cfg, f"best@{best[1]}", sc, f"retainKL={h['retain']['kl']:.3f} agree={h['retain']['agree']:.3f}"))
for r in rows: print(" | ".join(r))
