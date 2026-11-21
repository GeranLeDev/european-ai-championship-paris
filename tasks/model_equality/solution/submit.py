"""Submit a CSV (endpoint,family,model) for task 33-model-equality. Key from $MET_API_KEY. 1 submission / 30 min.
Run: MET_API_KEY=... python submit.py submission.csv
"""
import csv
import os
import sys

import requests

URL = "http://34.147.229.55/submit/33-model-equality"


def main():
    path = sys.argv[1]
    key = os.environ.get("MET_API_KEY", "").strip()
    if not key:
        sys.exit("set MET_API_KEY")
    rows = [r for r in csv.reader(open(path)) if r and r[0] != "endpoint"]
    eps = {r[0] for r in rows}
    assert len(rows) == 33 and eps == {f"ep-{i:02d}" for i in range(1, 34)}, "need exactly ep-01..ep-33"
    fam = {}
    for e, f, m in rows:                       # same model label => same family label
        assert fam.setdefault(m, f) == f, f"model {m} in two families"
    with open(path, "rb") as fh:
        r = requests.post(URL, headers={"X-API-Key": key}, files={"file": (os.path.basename(path), fh)},
                          timeout=(600, 600))
    print(r.status_code, r.text[:1000])


if __name__ == "__main__":
    main()
