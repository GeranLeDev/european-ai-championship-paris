"""Tokenizer fingerprint through the 1,024-input-token limit (family signal that no disguise can change).

Measured on ref (21/11): the server counts the input tokens with the endpoint's OWN tokenizer, on the raw message
(no chat template, no hidden system prompt). A request above the limit is rejected with HTTP 400 and is NOT counted.
All 9 ref endpoints (plain, fp8, system prompt, persona, watermark, T=1.0, LoRA 0.3/0.7/1.5) accept exactly 51
repetitions of UNITS["words"] and 35 of UNITS["mixed"]: that is SmolLM2's tokenizer computed locally (51 and 35).
Qwen2.5's tokenizer gives 53 and 40.

Method: for each endpoint and each text unit, send k repetitions with max_tokens=1, k going DOWN from --start one by
one. Rejections are free; the first accepted request ends the search. Cost: exactly 1 counted answer per endpoint and
per unit. The boundary k* (largest accepted k) is the fingerprint: same tokenizer <=> same k* for every unit.
If the very first request is accepted, k* is only known to be >= the start value (still a usable group label);
a rerun with a larger start costs 1 more answer. A rejection takes about 7 s: keep the start values close.

  MET_API_KEY=... python tokprobe.py --endpoints ref                       # free
  MET_API_KEY=... python tokprobe.py --endpoints ep --allow-scored         # 1 answer per endpoint and per unit
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from collect import BASE_URL, api_key, get_budget, resolve_endpoints, session_for
from met_common import HERE, read_jsonl

UNITS = {
    # ~20 tokens per repetition, split very differently by BPE vocabularies
    "words": "pneumonoultramicroscopic counterrevolutionaries 3.14159 naïve ",
    "mixed": "def f(x): return x**2 # 北京 Straße 1234567890 ",
}


def probe(sess, ep, unit, start, lock, fh, floor=8):
    k = start
    while k >= floor:
        body = {"model": ep, "max_tokens": 1, "messages": [{"role": "user", "content": UNITS[unit] * k}]}
        try:
            r = sess.post(f"{BASE_URL}/chat/completions", json=body, timeout=(20, 900))
        except Exception as exc:
            print(f"{ep} {unit} k={k}: network error ({type(exc).__name__}), retry in 30 s", flush=True)
            time.sleep(30)
            continue
        if r.status_code == 200:                      # counted: first accepted length, stop here
            rec = {"endpoint": ep, "unit": unit, "accepted": k, "exact": k < start, "t": time.time(),
                   "budget": r.json().get("budget")}
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            print(f"{ep} {unit}: largest accepted = {k}{'' if k < start else ' (>= start: rerun with a larger --start)'}", flush=True)
            return
        if r.status_code == 400:                      # too long: not counted
            k -= 1
        elif r.status_code == 429 and "Retry-After" in r.headers:
            time.sleep(int(r.headers["Retry-After"]))
        elif r.status_code == 429:
            print(f"{ep}: budget exhausted", flush=True)
            return
        elif r.status_code in (502, 503, 504):
            time.sleep(15)
        else:
            print(f"{ep} {unit} k={k}: HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return
    print(f"{ep} {unit}: nothing accepted down to {floor}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", default="ref")
    ap.add_argument("--allow-scored", action="store_true")
    ap.add_argument("--units", default="words,mixed")
    ap.add_argument("--start", default="words=56,mixed=44",
                    help="first (largest) number of repetitions tried per unit; SmolLM2 accepts 51 / 35, Qwen2.5 53 / 40")
    ap.add_argument("--out", default=None)
    ap.add_argument("--key-file", default=None)
    a = ap.parse_args()
    key = api_key(a)
    if not key:
        sys.exit("no API key")
    sess = session_for(key)
    eps = resolve_endpoints(a.endpoints, get_budget(sess))
    if any(e.startswith("ep-") for e in eps) and not a.allow_scored:
        sys.exit("refusing to query scored endpoints without --allow-scored")
    out = Path(a.out or HERE / "data" / f"{eps[0].split('-')[0]}_tok.jsonl")
    done = {(r["endpoint"], r["unit"]) for r in read_jsonl(out) if r.get("exact")}
    jobs = [(e, u) for u in a.units.split(",") for e in eps if (e, u) not in done]
    print(f"{len(jobs)} probes (1 counted answer each), {len(done)} already known -> {out}", flush=True)
    start = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.start.split(",")}
    lock = threading.Lock()
    with out.open("a") as fh, ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda j: probe(sess, j[0], j[1], start[j[1]], lock, fh), jobs))
    table = {}
    for r in read_jsonl(out):
        table.setdefault(r["endpoint"], {})[r["unit"]] = r["accepted"]
    groups = {}
    for e, d in sorted(table.items()):
        groups.setdefault(tuple(sorted(d.items())), []).append(e)
    for sig, members in groups.items():
        print(dict(sig), "->", ",".join(members))


if __name__ == "__main__":
    main()
