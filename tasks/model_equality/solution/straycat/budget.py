"""Budget restant par API (requête gratuite). Lit la clé dans le .env du dépôt de l'équipe, sans l'afficher."""
import json, sys, time
from pathlib import Path
import requests
ENV = Path("/Users/amil/Downloads/hackathon-watermark/european-ai-championship-paris/.env")
KEY = dict(l.strip().split("=", 1) for l in ENV.read_text().splitlines() if "=" in l)["HACKATHON_API_KEY"]
BASE = "http://34.147.229.55/33-model-equality/v1"
def snapshot():
    r = requests.get(f"{BASE}/models", headers={"X-API-Key": KEY}, timeout=60); r.raise_for_status()
    return {m["id"]: m["budget"]["used"] for m in r.json()["data"]}
a = snapshot(); time.sleep(int(sys.argv[1]) if len(sys.argv) > 1 else 0); b = snapshot()
ep = {k: v for k, v in b.items() if k.startswith("ep-")}; ref = {k: v for k, v in b.items() if k.startswith("ref-")}
print(f"API notées : {len(ep)}, questions déjà dépensées au total : {sum(ep.values())} (max par API : {max(ep.values())})")
print(f"API ref : {len(ref)}, réponses ref déjà prises : {sum(ref.values())}")
moved = {k: b[k] - a[k] for k in b if b[k] != a[k]}
print("activité pendant l'attente :", moved if moved else "aucune")
