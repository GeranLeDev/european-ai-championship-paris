"""Query collector for task 33-model-equality (cache to disk, resume, <=4 concurrent, budget accounting, dry-run).

SAFETY
  * By default only the free reference endpoints (ref-XX) can be queried. Querying the scored endpoints (ep-XX, 100
    answers each for the WHOLE competition, shared by the team) needs --allow-scored AND an explicit --endpoints list
    or 'ep'.
  * --dry-run never sends a POST (it may call GET /models, which is free, if a key is set and --no-network is absent).
  * The API key is read from $MET_API_KEY or --key-file; it is never written to disk by this script.

Examples
  python collect.py --endpoints ref --dry-run --no-network            # plan only, offline
  MET_API_KEY=... python collect.py --endpoints ref --out data/ref_answers.jsonl
  MET_API_KEY=... python collect.py --endpoints ep --allow-scored --out data/ep_answers.jsonl --max-per-endpoint 100
  MET_API_KEY=... python collect.py probe          # capability probe on ref-01 only (logprobs, prefill, n, usage)
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from met_common import HERE, read_jsonl

SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"
BASE_URL = os.environ.get("MET_BASE_URL") or f"{SERVER_URL}/{TASK_ID}/v1"   # MET_BASE_URL: local mock server (tests)
MAX_PARALLEL = 4
MAX_TOKENS = 256
STOP_FILE = Path(os.environ.get("MET_STOP_FILE") or HERE / "data" / "STOP")          # `touch data/STOP` = stop cleanly (no paid answer is lost)


def api_key(args):
    key = os.environ.get("MET_API_KEY", "").strip()
    if not key and args.key_file:
        key = Path(args.key_file).read_text().strip()
    return key


def session_for(key):
    import requests
    s = requests.Session()
    s.headers["X-API-Key"] = key
    return s


def get_budget(sess):
    r = sess.get(f"{BASE_URL}/models", timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"GET /models HTTP {r.status_code}: {r.text[:300]}")
    return {m["id"]: m["budget"]["limit"] - m["budget"]["used"] for m in r.json()["data"]}


def resolve_endpoints(spec, budget):
    if spec in ("ref", "ep"):
        if budget:
            return sorted(e for e in budget if e.startswith(spec + "-"))
        n = 9 if spec == "ref" else 33
        return [f"{spec}-{i:02d}" for i in range(1, n + 1)]
    return [e.strip() for e in spec.split(",") if e.strip()]


def plan(prompts, endpoints, done, cap, left, order="repeat"):
    """Round-robin by repeat then prompt then endpoint (a run that stops early stays balanced)."""
    used = {e: sum(1 for (ep, _, _) in done if ep == e) for e in endpoints}
    jobs, skipped = [], 0
    max_rep = max(p["repeats"] for p in prompts)
    if order == "prompt":       # prompt-major: all repeats of prompt 1, then prompt 2... (any prefix is analysable)
        seq = [(rep, p) for p in prompts for rep in range(p["repeats"])]
    else:
        seq = [(rep, p) for rep in range(max_rep) for p in prompts if rep < p["repeats"]]
    for rep, p in seq:
        for e in endpoints:
            if (e, p["id"], rep) in done:
                continue
            if used[e] < cap and (left is None or left.get(e, 0) > 0):
                used[e] += 1
                if left is not None:
                    left[e] -= 1
                jobs.append((e, p["id"], rep, p))
            else:
                skipped += 1
    return jobs, skipped, used


def run_jobs(sess, jobs, out_path, parallel, extra):
    lock, exhausted, counter = threading.Lock(), set(), [0]
    out = open(out_path, "a", encoding="utf-8")

    def one(job):
        ep, pid, rep, p = job
        tries = 0
        while ep not in exhausted and tries < 50:
            if STOP_FILE.exists():            # graceful stop: in-flight requests finish and are saved, no new one starts
                return
            tries += 1
            # a prompt is either one user message ("prompt") or a full user/assistant conversation ("messages")
            body = {"model": ep, "max_tokens": p.get("max_tokens", MAX_TOKENS),
                    "messages": p.get("messages") or [{"role": "user", "content": p["prompt"]}]}
            body.update(extra)
            try:
                # connect timeout short (nothing is counted if we cannot connect), read timeout very long: a reply that
                # the server has already counted must not be dropped by an impatient client.
                t_send = time.time()
                r = sess.post(f"{BASE_URL}/chat/completions", json=body, timeout=(20, 900))
                latency = time.time() - t_send
            except Exception as exc:  # network
                print(f"{ep} {pid} r{rep}: network error ({type(exc).__name__}), retry in 30 s", flush=True)
                time.sleep(30)
                continue
            if r.status_code == 200:
                try:
                    b = r.json()
                    ch = b["choices"][0]
                    ch["message"]["content"]
                except Exception:             # counted by the server but unreadable: keep the raw body, do not re-ask
                    with lock:
                        out.write(json.dumps({"endpoint": ep, "id": pid, "repeat": rep, "response": "",
                                              "raw": r.text[:5000], "bad": True, "t": time.time()}) + "\n")
                        out.flush()
                        os.fsync(out.fileno())
                    return
                rec = {"endpoint": ep, "id": pid, "repeat": rep, "response": ch["message"]["content"],
                       "finish_reason": ch.get("finish_reason"), "usage": b.get("usage"), "budget": b.get("budget"),
                       "extra_keys": sorted(set(b) - {"choices", "usage", "budget"}), "t": time.time(),
                       "latency": round(latency, 3), "tries": tries}
                if "logprobs" in ch and ch["logprobs"]:
                    rec["logprobs"] = ch["logprobs"]
                with lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                    counter[0] += 1
                    if counter[0] % 20 == 0 or counter[0] == len(jobs):
                        print(f"{counter[0]}/{len(jobs)} ({ep} budget {b.get('budget')})", flush=True)
                return
            ctype = r.headers.get("content-type", "")
            detail = r.json().get("detail", r.text[:300]) if ctype.startswith("application/json") else r.text[:300]
            if r.status_code == 429 and "Retry-After" in r.headers:
                time.sleep(int(r.headers["Retry-After"]))
            elif r.status_code == 429:
                with lock:
                    exhausted.add(ep)
                print(f"{ep}: budget exhausted: {detail}", flush=True)
            elif r.status_code in (502, 503, 504):
                time.sleep(15)
            else:
                print(f"{ep} {pid}: HTTP {r.status_code}: {detail}", file=sys.stderr, flush=True)
                return

    try:
        with ThreadPoolExecutor(max_workers=min(parallel, MAX_PARALLEL)) as pool:
            list(pool.map(one, jobs))
    finally:
        out.close()


def cmd_collect(args):
    prompts = read_jsonl(args.prompts)
    key = api_key(args)
    budget = None
    if key and not args.no_network:
        budget = get_budget(session_for(key))
    endpoints = resolve_endpoints(args.endpoints, budget)
    scored = [e for e in endpoints if e.startswith("ep-")]
    if scored and not args.allow_scored:
        sys.exit(f"refusing to query scored endpoints {scored[:3]}... without --allow-scored")
    done = {(r["endpoint"], r["id"], r["repeat"]) for r in read_jsonl(args.out)}
    left = None if budget is None else {e: budget.get(e, 0) for e in endpoints}
    if left is not None and any(e.startswith("ref-") for e in endpoints):
        left = {e: (10**9 if e.startswith("ref-") else v) for e, v in left.items()}   # ref has no limit
    jobs, skipped, used = plan(prompts, endpoints, done, args.max_per_endpoint, left, args.order)
    per_ep = {}
    for j in jobs:
        per_ep[j[0]] = per_ep.get(j[0], 0) + 1
    print(f"{len(prompts)} prompts ({sum(p['repeats'] for p in prompts)} answers/endpoint planned), "
          f"{len(endpoints)} endpoints, {len(done)} cached in {args.out}, {len(jobs)} to send, {skipped} skipped (cap/budget)")
    if budget:
        print("server budget left:", {e: budget.get(e) for e in endpoints})
    print("to send per endpoint:", per_ep)
    if args.dry_run:
        print("DRY RUN: nothing sent.")
        return
    if not key:
        sys.exit("no API key: set MET_API_KEY (or --key-file)")
    if scored:
        print(f"!!! about to spend up to {sum(per_ep.get(e, 0) for e in scored)} answers of the SCORED budget", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    run_jobs(session_for(key), jobs, args.out, args.parallel, json.loads(args.extra) if args.extra else {})
    print("done ->", args.out)


def cmd_probe(args):
    """Capability probe, ref-01 only (free): what does the server return / accept?"""
    key = api_key(args)
    if not key:
        sys.exit("no API key")
    sess = session_for(key)
    ep = args.probe_endpoint
    if not ep.startswith("ref-"):
        sys.exit("probe only runs on ref-XX endpoints")
    print("budget:", get_budget(sess))
    tests = {
        "plain": {},
        "logprobs": {"logprobs": True, "top_logprobs": 5},
        "n2": {"n": 2},
        "prefill": {"messages": [{"role": "user", "content": "Name a colour."},
                                 {"role": "assistant", "content": "The colour I pick is"}]},
        "multiturn": {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
                                   {"role": "user", "content": "Name a colour."}]},
        "system": {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Name a colour."}]},
        "max_tokens_1": {"max_tokens": 1},
    }
    for name, over in tests.items():
        body = {"model": ep, "max_tokens": 32, "messages": [{"role": "user", "content": "Name a colour."}]}
        body.update(over)
        r = sess.post(f"{BASE_URL}/chat/completions", json=body, timeout=120)
        txt = r.text[:600].replace("\n", " ")
        print(f"--- {name}: HTTP {r.status_code}: {txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="collect", choices=["collect", "probe"])
    ap.add_argument("--prompts", default=str(HERE / "prompts.jsonl"))
    ap.add_argument("--out", default=str(HERE / "data" / "ref_answers.jsonl"))
    ap.add_argument("--endpoints", default="ref", help="'ref', 'ep' or comma list")
    ap.add_argument("--allow-scored", action="store_true")
    ap.add_argument("--max-per-endpoint", type=int, default=100)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--order", default="repeat", choices=["repeat", "prompt"],
                    help="'prompt' = send all repeats of prompt 1 first, then prompt 2... (staged budget)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--extra", default=None, help="JSON merged into each request body (ref probing only)")
    ap.add_argument("--probe-endpoint", default="ref-01")
    args = ap.parse_args()
    if args.extra and args.allow_scored:
        sys.exit("--extra is for ref probing only")
    (cmd_probe if args.cmd == "probe" else cmd_collect)(args)


if __name__ == "__main__":
    main()
