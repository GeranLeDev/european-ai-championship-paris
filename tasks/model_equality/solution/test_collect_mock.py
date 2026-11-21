"""Offline test of collect.py against a local mock of the API (never touches the real server).

Checks: stops exactly at --max-per-endpoint, 429+Retry-After and 503 are retried without double spending, a killed run
resumes without re-asking saved answers, and server-side 'used' == answers saved on disk.
Run: python test_collect_mock.py
"""
import json, os, random, signal, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
USED = {f"ep-{i:02d}": 0 for i in range(1, 4)}
LOCK = threading.Lock()
RNG = random.Random(0)
ACTIVE = [0]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj, headers=None):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        with LOCK:
            self._send(200, {"data": [{"id": e, "budget": {"limit": 100, "used": u}} for e, u in USED.items()]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        ep = body["model"]
        with LOCK:
            x = RNG.random()
            if ACTIVE[0] >= 4 or x < 0.15:
                return self._send(429, {"detail": "too many concurrent"}, {"Retry-After": "1"})
            if x < 0.25:
                return self._send(503, {"detail": "busy"})
            if USED[ep] >= 100:
                return self._send(429, {"detail": "budget exhausted"})
            ACTIVE[0] += 1
            USED[ep] += 1                      # counted BEFORE the answer is sent, like a real server
            u = USED[ep]
        time.sleep(RNG.uniform(0.05, 0.3))
        with LOCK:
            ACTIVE[0] -= 1
        self._send(200, {"choices": [{"message": {"content": f"answer {u}"}, "finish_reason": "stop"}],
                         "budget": {"used": u, "limit": 100}})


def run(args, env, kill_after=None):
    p = subprocess.Popen([sys.executable, str(HERE / "collect.py")] + args, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    if kill_after:
        time.sleep(kill_after)
        p.send_signal(signal.SIGKILL)
    out = p.communicate()[0]
    return out


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    assert "127.0.0.1" in url
    tmp = Path(tempfile.mkdtemp())
    stop = tmp / "STOP"                 # private stop file: never interferes with a real collection
    env = dict(os.environ, MET_BASE_URL=url, MET_API_KEY="mock", MET_STOP_FILE=str(stop))
    prompts = tmp / "p.jsonl"
    prompts.write_text("".join(json.dumps({"id": f"q{i}", "kind": "short", "repeats": 2, "prompt": "x"}) + "\n"
                               for i in range(50)))
    out = tmp / "ans.jsonl"
    base = ["--endpoints", "ep", "--allow-scored", "--prompts", str(prompts), "--out", str(out), "--order", "prompt"]

    def saved():
        rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
        keys = [(r["endpoint"], r["id"], r["repeat"]) for r in rows]
        assert len(keys) == len(set(keys)), "duplicate answers on disk"
        return rows

    run(base + ["--max-per-endpoint", "20"], env, kill_after=2.0)          # hard kill mid-run
    n1 = len(saved())
    lost = sum(USED.values()) - n1
    print(f"after SIGKILL: {n1} saved, server used {sum(USED.values())} -> {lost} in-flight answers lost (<= 4 expected)")
    assert 0 <= lost <= 4
    run(base + ["--max-per-endpoint", "20"], env)                          # resume to the cap
    rows = saved()
    per = {e: sum(r["endpoint"] == e for r in rows) for e in USED}
    print("after resume: saved per endpoint", per, "server used", USED)
    assert all(v == 20 for v in per.values()), "cap not respected"
    assert sum(USED.values()) == 60 + lost, "an answer was paid twice or not saved"
    ids = {(r["id"], r["repeat"]) for r in rows if r["endpoint"] == "ep-01"}
    assert ids == {(f"q{i}", r) for i in range(10) for r in range(2)}, "pass 1 is not 'first 10 prompts x 2 repeats'"
    txt = run(base + ["--max-per-endpoint", "20"], env)                    # nothing left to do
    assert "0 to send" in txt and sum(USED.values()) == 60 + lost
    stop.touch()                                       # graceful stop: nothing new is sent
    try:
        before = sum(USED.values())
        run(base + ["--max-per-endpoint", "40"], env)
        assert sum(USED.values()) == before and len(saved()) == 60
    finally:
        stop.unlink()
    run(base + ["--max-per-endpoint", "200"], env)                         # run into the server budget (100)
    rows = saved()
    print("full budget: saved", len(rows), "server used", USED)
    assert sum(USED.values()) == 300 and len(rows) == 300 - lost
    print("OK")


if __name__ == "__main__":
    main()
