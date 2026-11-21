"""Shared helpers: prompt/answer IO, answer parsing/canonicalisation, official-style scoring."""
import csv
import itertools
import json
import math
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_prompts(path=HERE / "prompts.jsonl"):
    path = Path(path)
    prompts = read_jsonl(path)
    items = json.loads(path.with_name(path.stem + "_items.json").read_text())
    return prompts, items


# ---------------------------------------------------------------- parsing
_LINE = re.compile(r"^\s*\**\s*(\d{1,2})\s*[\.\):]\s*(.*)$")


def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def canon(ans, typ, fillers=()):
    """Canonical form of one short answer: first number for numeric items, first content word otherwise.
    `fillers`: endpoint-specific filler words to skip (persona interjections like 'arr')."""
    s = _strip_accents(ans).lower()
    s = re.sub(r"\*+", "", s)
    for sep in ("=", "->", "→", ":", " - ", " – "):          # "173 + 762 = 935", "friend -> ami", "Answer: x"
        if sep in s and s.rsplit(sep, 1)[1].strip():
            s = s.rsplit(sep, 1)[1]
    if typ == "n":
        m = re.search(r"-?\d[\d,]*(?:\.\d+)?", s)
        return m.group(0).replace(",", "") if m else "<none>"
    s = re.sub(r"^[\s\-:]*(?:answer|the answer is|it is|it's|is)\s*[:\-]?\s*", "", s)
    s = re.split(r"[,(;!?]", s)[0]           # "Paris, matey!" -> "paris" ; "Nickel (Ni)" -> "nickel"
    words = [w.strip("'") for w in re.findall(r"[a-z0-9']+", s) if w.strip("'")]
    stop = {"the", "a", "an", "le", "la", "les", "der", "die", "das", "l'", "un", "une", "ein", "eine"}
    words = [w for w in words if w not in stop and w not in fillers]
    return words[-1] if words else "<none>"   # last word of the first segment: "New Delhi" ~ "Delhi"


_QLINE = re.compile(r"^\s*\**\s*(?:question|q)\s*(\d{1,2})\s*[\.\):]?\s*(.*)$", re.I)
_ALINE = re.compile(r"^\s*\**\s*answer\s*\**\s*[:\-]\s*(.*)$", re.I)


def quiz_lines(response, n):
    """{question number: answer text}. Accepts '3. Paris' and the verbose 'Question 3: ...' / 'Answer: Paris'."""
    found, cur = {}, None
    for line in response.splitlines():
        m = _LINE.match(line)
        if m:
            k = int(m.group(1))
            if 1 <= k <= n and k not in found:
                found[k] = m.group(2)
            cur = None
            continue
        m = _QLINE.match(line)
        if m:
            cur = int(m.group(1))
            continue
        m = _ALINE.match(line)
        if m and cur is not None and 1 <= cur <= n:
            found[cur] = m.group(1)
            cur = None
    if not found:                                   # unnumbered reply: positional only if exactly n lines
        lines = [l.strip() for l in response.splitlines() if l.strip()]
        if len(lines) == n:
            found = {k + 1: v for k, v in enumerate(lines)}
    return found


def endpoint_fillers(responses, frac=0.15):
    """Words that appear in more than `frac` of an endpoint's answer lines (across different questions) are
    style fillers (persona 'arr', 'answer', 'sir'...), not answers."""
    from collections import Counter
    cnt, n = Counter(), 0
    for r in responses:
        for v in quiz_lines(r, 99).values():
            n += 1
            cnt.update(set(re.findall(r"[a-z']+", _strip_accents(v).lower())))
    return {w for w, c in cnt.items() if n and c / n > frac}


def parse_quiz(response, items, fillers=()):
    """Returns a list (len(items)) of canonical answers, '<miss>' when the line is absent."""
    found = quiz_lines(response, len(items))
    return [canon(found[j + 1], it["type"], fillers) if (j + 1) in found else "<miss>" for j, it in enumerate(items)]


# ---------------------------------------------------------------- scoring (task formula)
def mcc(tp, fp, fn, tn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


def read_labels(path):
    rows = [r for r in csv.reader(open(path, newline="", encoding="utf-8-sig")) if any(c.strip() for c in r)]
    if rows and [c.strip() for c in rows[0][:3]] == ["endpoint", "family", "model"]:
        rows = rows[1:]
    return {r[0].strip(): (r[1].strip(), r[2].strip()) for r in rows}


def task_score(truth, guess):
    """0.1 * MCC_family (all pairs) + 0.9 * MCC_model (pairs inside one TRUE family). truth/guess: ep -> (fam, model)."""
    eps = sorted(truth)
    cf = [0, 0, 0, 0]
    cm = [0, 0, 0, 0]
    for a, b in itertools.combinations(eps, 2):
        tf = truth[a][0] == truth[b][0]
        gf = guess[a][0] == guess[b][0]
        cf[_cell(tf, gf)] += 1
        if tf:
            tm = truth[a] == truth[b]
            gm = guess[a] == guess[b]
            cm[_cell(tm, gm)] += 1
    mf, mm = mcc(*cf), mcc(*cm)
    return {"score": 0.1 * mf + 0.9 * mm, "mcc_family": mf, "mcc_model": mm,
            "family_conf(tp,fp,fn,tn)": cf, "model_conf(tp,fp,fn,tn)": cm}


def _cell(t, g):
    return 0 if (t and g) else 1 if (g and not t) else 2 if (t and not g) else 3


def write_submission(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["endpoint", "family", "model"])
        w.writerows(rows)
