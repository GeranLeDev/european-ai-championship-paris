"""Side-channel probes on the FREE ref endpoints only (refuses anything else).

  limit   : send a prompt that is too long -> the request is rejected (not counted). Does the error message give the
            token count, and does the accepted/rejected boundary depend on the endpoint (tokenizer fingerprint)?
  latency : same prompt, one request at a time, max_tokens fixed -> seconds per request and per character.

Run: MET_API_KEY=... python side_probe.py limit|latency [--endpoints ref-01,ref-02]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import requests

from collect import BASE_URL

WORD = "pneumonoultramicroscopic counterrevolutionaries 3.14159 naïve こんにちは "


def post(sess, ep, content, max_tokens):
    assert ep.startswith("ref-"), "side probes run on ref endpoints only"
    t0 = time.time()
    r = sess.post(f"{BASE_URL}/chat/completions", timeout=(20, 600),
                  json={"model": ep, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]})
    return r, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["limit", "latency"])
    ap.add_argument("--endpoints", default=",".join(f"ref-{i:02d}" for i in range(1, 10)))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="data/ref_side.jsonl")
    a = ap.parse_args()
    key = os.environ.get("MET_API_KEY", "").strip()
    if not key:
        sys.exit("no API key")
    eps = [e for e in a.endpoints.split(",") if e]
    if any(not e.startswith("ref-") for e in eps):
        sys.exit("ref endpoints only")
    sess = requests.Session()
    sess.headers["X-API-Key"] = key
    out = open(a.out, "a")
    if a.cmd == "limit":
        for ep in eps[:3]:
            for k in (40, 120, 400):                       # repetitions of WORD: short, borderline, far too long
                r, dt = post(sess, ep, WORD * k, 1)
                body = r.text[:400].replace("\n", " ")
                print(f"{ep} x{k} ({len(WORD) * k} chars): HTTP {r.status_code} in {dt:.1f}s: {body}", flush=True)
                out.write(json.dumps({"probe": "limit", "endpoint": ep, "k": k, "status": r.status_code, "body": r.text[:1000]}) + "\n")
    else:
        prompt = "Count from 1 to 300. Write only the numbers, separated by spaces."
        for rep in range(a.reps):
            for ep in eps:
                r, dt = post(sess, ep, prompt, 256)
                if r.status_code != 200:
                    print(ep, "HTTP", r.status_code, r.text[:200], flush=True)
                    time.sleep(int(r.headers.get("Retry-After", 5)))
                    continue
                ch = r.json()["choices"][0]
                txt = ch["message"]["content"]
                rec = {"probe": "latency", "endpoint": ep, "rep": rep, "latency": round(dt, 3), "chars": len(txt),
                       "finish": ch.get("finish_reason"), "response": txt}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{ep} rep {rep}: {dt:6.2f}s {len(txt):5d} chars finish={ch.get('finish_reason')} "
                      f"{1000 * dt / max(len(txt), 1):.1f} ms/char", flush=True)


if __name__ == "__main__":
    main()
