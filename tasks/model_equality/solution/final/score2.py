"""Temperature-invariant identity test against a local public base model.
For each endpoint: fit the sampling temperature T that maximises the likelihood of its tokens under softmax(logits/T),
then R = mean over tokens of [-log q_T(token) - H(q_T)]. Samples of the base model at ANY temperature give R ~ 0;
a model with other weights gives R > 0 (extra cross-entropy)."""
import json, sys, math, torch, html, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name, famname = sys.argv[1], sys.argv[2]
extra = sys.argv[3] if len(sys.argv) > 3 else None       # jsonl of extra answers (collect.py format) + prompts file
D = json.load(open('ep.json')); fam = D['fam']; free = D['free']; meta = D['meta']
eps = sorted(e for e in fam if fam[e][0] == famname)
tok = AutoTokenizer.from_pretrained(model_name)
dev = 'mps'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(dev).eval()
TS = np.round(np.arange(0.3, 1.55, 0.05), 2)
items = []   # (endpoint, promptkey, prompt, answer, maxtok)
for p in free:
    prompt = html.unescape('\n'.join(meta[p]['prompt']).strip())
    for e in eps:
        if free[p].get(e): items.append((e, p, prompt, html.unescape(free[p][e]), meta[p]['max']))
if extra:
    for spec in extra.split(','):
        ans, pr = spec.split(':')
        P = {json.loads(l)['id']: json.loads(l)['prompt'] for l in open(pr)}
        for l in open(ans):
            r = json.loads(l)
            if r['endpoint'] in eps and r['response']: items.append((r['endpoint'], r['id'] + '#' + str(r['repeat']), P[r['id']], r['response'], 256))
res = {}
cache = {}
for e, p, prompt, y, mx in items:
    if prompt not in cache:
        pre = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        cache[prompt] = tok(pre, add_special_tokens=False)['input_ids']
    pre_ids = cache[prompt]
    ids = tok(y, add_special_tokens=False)['input_ids'][:mx]
    if len(ids) < 2: continue
    x = torch.tensor([pre_ids + ids], device=dev)
    with torch.no_grad():
        lg = model(x).logits[0, len(pre_ids) - 1:-1].cpu().double()
    t = torch.tensor(ids)
    ll = np.zeros((len(TS), len(ids))); H = np.zeros((len(TS), len(ids)))
    for k, T in enumerate(TS):
        lp = torch.log_softmax(lg / T, -1)
        ll[k] = lp[torch.arange(len(ids)), t].numpy()
        H[k] = -(lp.exp() * lp).sum(-1).numpy()
    res.setdefault(e, []).append((p, ll, H))
import pickle; pickle.dump(res, open(f'score2_{famname}.pkl', 'wb'))
print('endpoint note           tokens   T_hat   R=xent-H at T_hat (se)    z     NLL@T=1')
for e in eps:
    if e not in res: continue
    LL = np.concatenate([r[1] for r in res[e]], 1); HH = np.concatenate([r[2] for r in res[e]], 1)
    k = int(np.argmax(LL.sum(1)))
    per = np.array([(-r[1][k] - r[2][k]).sum() for r in res[e]]); n = np.array([r[1].shape[1] for r in res[e]])
    R = per.sum() / n.sum()
    # bootstrap over answers
    rng = np.random.default_rng(0); bs = []
    for _ in range(300):
        i = rng.integers(0, len(per), len(per)); bs.append(per[i].sum() / n[i].sum())
    se = np.std(bs)
    k1 = list(TS).index(1.0)
    print(f"{e} {fam[e][1][:6]:6s} {fam[e][2][:12]:12s} {n.sum():6d}   {TS[k]:.2f}   {R:+.4f} ({se:.4f})   {R/se:+5.1f}   {-LL[k1].mean():.3f}")
