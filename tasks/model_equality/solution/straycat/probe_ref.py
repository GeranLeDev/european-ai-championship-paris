"""Test sur les 9 API de ref (gratuites) : la même question deux fois donne-t-elle le même texte ? Combien de temps ?
Ecrit chaque réponse dans probe_ref.jsonl (endpoint, prompt, essai, texte, secondes)."""
import json, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
ENV = Path("/Users/amil/Downloads/hackathon-watermark/european-ai-championship-paris/.env")
KEY = dict(l.strip().split("=", 1) for l in ENV.read_text().splitlines() if "=" in l)["HACKATHON_API_KEY"]
BASE = "http://34.147.229.55/33-model-equality/v1"
PROMPTS = ["Write exactly one sentence describing the ocean at night.",
           "List the first ten prime numbers, separated by commas, and nothing else."]
REPEATS, MAXTOK = 2, int(sys.argv[1]) if len(sys.argv) > 1 else 48
ENDPOINTS = [f"ref-{i:02d}" for i in range(1, 10)]
out = open("probe_ref.jsonl", "a"); lock = threading.Lock(); s = requests.Session(); s.headers["X-API-Key"] = KEY
def one(job):
    ep, pi, rep = job
    for _ in range(20):
        t0 = time.time()
        r = s.post(f"{BASE}/chat/completions", timeout=180, json={"model": ep, "max_tokens": MAXTOK, "messages": [{"role": "user", "content": PROMPTS[pi]}]})
        dt = time.time() - t0
        if r.status_code == 200:
            txt = r.json()["choices"][0]["message"]["content"]
            with lock: out.write(json.dumps(dict(endpoint=ep, prompt=pi, repeat=rep, text=txt, seconds=round(dt, 2), max_tokens=MAXTOK)) + "\n"); out.flush()
            return
        if r.status_code == 429 and "Retry-After" in r.headers: time.sleep(int(r.headers["Retry-After"])); continue
        if r.status_code in (502, 503, 504): time.sleep(10); continue
        print(ep, r.status_code, r.text[:200]); return
jobs = [(ep, pi, rep) for rep in range(REPEATS) for pi in range(len(PROMPTS)) for ep in ENDPOINTS]
t0 = time.time()
with ThreadPoolExecutor(4) as pool: list(pool.map(one, jobs))
print(f"{len(jobs)} requêtes en {time.time() - t0:.0f} s")
