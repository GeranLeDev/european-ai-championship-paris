"""Submit a file to the leaderboard. Key read from <repo>/.env (HACKATHON_API_KEY=...).
Usage: python scripts/submit.py <task_id> <file>
Task ids: 34-watermark-localisation-image (.pt), 33-model-equality (.csv, 1 per 30 min), 31-unlearning (.pt state_dict)."""
import sys, requests
key = dict(l.strip().split("=",1) for l in open(__import__("pathlib").Path(__file__).resolve().parents[1] / ".env") if "=" in l)["HACKATHON_API_KEY"]
task, path = sys.argv[1], sys.argv[2]
with open(path, "rb") as f:
    r = requests.post(f"http://34.147.229.55/submit/{task}", headers={"X-API-Key": key}, files={"file": (path.split("/")[-1], f)}, timeout=(60, 900))
try: b = r.json()
except ValueError: b = r.text[:300]
print(task, "HTTP", r.status_code, b)
