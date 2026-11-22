"""Generic version of score2: answers jsonl (collect.py format) + prompts jsonl, local model, optional endpoint prefix."""
import json, sys, torch, pickle, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name, ans, prm, outp = sys.argv[1:5]
kinds = sys.argv[5].split(',') if len(sys.argv) > 5 else None
P = {json.loads(l)['id']: json.loads(l) for l in open(prm)}
tok = AutoTokenizer.from_pretrained(model_name); dev = 'mps'
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(dev).eval()
TS = np.round(np.arange(0.3, 1.55, 0.05), 2)
res, cache = {}, {}
for l in open(ans):
    r = json.loads(l)
    if r['id'] not in P or not r['response']: continue
    if kinds and P[r['id']].get('kind') not in kinds: continue
    prompt = P[r['id']]['prompt']
    if prompt not in cache:
        cache[prompt] = tok(tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False), add_special_tokens=False)['input_ids']
    pre = cache[prompt]; ids = tok(r['response'], add_special_tokens=False)['input_ids'][:256]
    if len(ids) < 2: continue
    with torch.no_grad():
        lg = model(torch.tensor([pre + ids], device=dev)).logits[0, len(pre) - 1:-1].cpu().double()
    t = torch.tensor(ids); ll = np.zeros((len(TS), len(ids))); H = np.zeros((len(TS), len(ids)))
    for k, T in enumerate(TS):
        lp = torch.log_softmax(lg / T, -1); ll[k] = lp[torch.arange(len(ids)), t].numpy(); H[k] = -(lp.exp() * lp).sum(-1).numpy()
    res.setdefault(r['endpoint'], []).append((r['id'], ll, H))
pickle.dump(res, open(outp, 'wb'))
def that(rs):
    L = np.array([sum(x[1][k].sum() for x in rs) for k in range(len(TS))]); k = min(max(int(np.argmax(L)), 1), len(TS) - 2)
    a, b, c = np.polyfit(TS[k-1:k+2], L[k-1:k+2], 2); return -b / (2 * a), k
print('endpoint  tokens  T_hat (se)   R (se)   NLL@T=1')
for e in sorted(res):
    rs = res[e]; t, k = that(rs); rng = np.random.default_rng(0)
    bt = [that([rs[i] for i in rng.integers(0, len(rs), len(rs))])[0] for _ in range(100)]
    per = np.array([(-x[1][k] - x[2][k]).sum() for x in rs]); n = np.array([x[1].shape[1] for x in rs])
    bR = []
    for _ in range(200):
        i = rng.integers(0, len(per), len(per)); bR.append(per[i].sum() / n[i].sum())
    k1 = list(TS).index(1.0)
    print(f"{e}  {n.sum():6d}  {t:.3f} ({np.std(bt):.3f})  {per.sum()/n.sum():+.4f} ({np.std(bR):.4f})  {-sum(x[1][k1].sum() for x in rs)/n.sum():.3f}")
