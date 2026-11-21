"""Local simulator of the 33-model-equality deployment kinds with small HF models (MPS/CPU).

Builds simulated endpoints (same answer-file format as collect.py) + a truth CSV:
  different MODEL : plain base, fp8 weights (e4m3, per-channel), int4 weights (group 64), trained LoRA at strength s
  SAME model      : system prompt, persona, KGW watermark, decoding (temperature 1.0, greedy, beam 4)
Two families: Qwen2.5-0.5B-Instruct (A) and SmolLM2-360M-Instruct (B).

Run (from solution/):
  ../../../repos/model-equality-testing/.venv/bin/python sim.py --out data/sim_answers.jsonl
Options: --default-temp 0.7 (decoding of the plain endpoints), --repeats-scale 1, --only A
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from met_common import HERE, read_jsonl

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEV == "mps" else torch.float32
FAMILIES = {"A": "Qwen/Qwen2.5-0.5B-Instruct", "B": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "C": "Qwen/Qwen2.5-1.5B-Instruct"}
SYS_PROMPT = "You are a helpful customer support assistant for Contoso Bank. Keep answers short and professional."
PERSONA = "You are Captain Barnaby, an old pirate. Always talk like a pirate, with lots of 'Arr' and 'matey'."
LORA_STYLE = ("You are an upbeat assistant. You always answer in a cheerful tone, start with an enthusiastic "
              "interjection, and end with a fun fact.")

# (kind, model-label suffix, weights, system, decoding, watermark)
VARIANTS = {
    "A": [("plain", "S", None, None, "default", False), ("fp8", "Q8", "fp8", None, "default", False),
          ("int4", "Q4", "int4", None, "default", False), ("lora0.3", "L03", ("lora", 0.3), None, "default", False),
          ("lora0.7", "L07", ("lora", 0.7), None, "default", False), ("lora1.5", "L15", ("lora", 1.5), None, "default", False),
          ("sysprompt", "S", None, SYS_PROMPT, "default", False), ("persona", "S", None, PERSONA, "default", False),
          ("watermark", "S", None, None, "default", True), ("temp1.0", "S", None, None, "t1", False),
          ("greedy", "S", None, None, "greedy", False), ("beam", "S", None, None, "beam", False)],
    "B": [("plain", "S", None, None, "default", False), ("fp8", "Q8", "fp8", None, "default", False),
          ("lora1.0", "L10", ("lora", 1.0), None, "default", False), ("sysprompt", "S", None, SYS_PROMPT, "default", False),
          ("watermark", "S", None, None, "default", True), ("greedy", "S", None, None, "greedy", False)],
    # stronger model, mirrors the practice set (ref-01..09) + greedy
    "C": [("plain", "S", None, None, "default", False), ("fp8", "Q8", "fp8", None, "default", False),
          ("lora0.3", "L03", ("lora", 0.3), None, "default", False), ("lora0.7", "L07", ("lora", 0.7), None, "default", False),
          ("lora1.5", "L15", ("lora", 1.5), None, "default", False),
          ("sysprompt", "S", None, SYS_PROMPT, "default", False), ("persona", "S", None, PERSONA, "default", False),
          ("watermark", "S", None, None, "default", True), ("temp1.0", "S", None, None, "t1", False),
          ("greedy", "S", None, None, "greedy", False)],
}


class KGW(LogitsProcessor):
    """Kirchenbauer et al. soft watermark: green list (gamma) seeded by previous token, +delta on green logits."""

    def __init__(self, vocab, gamma=0.25, delta=2.0, key=15485863):
        self.gamma, self.delta, self.key = gamma, delta, key
        self.v = torch.arange(vocab, device=DEV, dtype=torch.int64)

    def __call__(self, input_ids, scores):
        prev = input_ids[:, -1:].to(torch.int64)
        h = (self.v[None, : scores.shape[1]] * 2654435761 + prev * 40503 + self.key) % 2147483647
        h = (h * 1103515245 + 12345) % 2147483647
        green = (h % 1000) < int(self.gamma * 1000)
        return scores + self.delta * green.to(scores.dtype)


# ------------------------------------------------------------------ weight transforms
def target_linears(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in n]


@torch.no_grad()
def quantize_(model, kind):
    for _, m in target_linears(model):
        w = m.weight.data.float()
        if kind == "fp8":
            s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 448.0
            q = (w / s).to(torch.float8_e4m3fn).to(torch.float32) * s
        else:  # int4 symmetric, groups of 64 along the input dim
            o, i = w.shape
            g = w.reshape(o, i // 64, 64)
            s = g.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / 7.0
            q = (torch.round(g / s).clamp(-8, 7) * s).reshape(o, i)
        m.weight.data.copy_(q.to(m.weight.dtype))


class LoRALinear(torch.nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) * 0.01)
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.t() @ self.B.t()) * self.scale).to(x.dtype)


LORA_TARGETS = ("q_proj", "v_proj", "o_proj", "up_proj", "down_proj")


def chat_ids(tok, user, system=None, gen=True):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=False)


def train_lora(model, tok, fam, cache):
    """Trains a small LoRA on self-generated 'upbeat style' answers. Returns {module_name: delta_W} (strength 1)."""
    if cache.exists():
        return torch.load(cache)
    topics = ["coffee", "the moon", "volcanoes", "saving money", "learning Spanish", "running", "jazz", "bees",
              "the Roman empire", "recycling", "chess", "sleep", "bread baking", "electric cars", "rainbows",
              "photosynthesis", "the internet", "dinosaurs", "meditation", "gardening", "sharks", "glaciers",
              "time management", "the piano", "black holes", "vaccines", "origami", "tea", "deserts", "robots"]
    forms = ["Tell me about {}.", "Give me one tip about {}.", "Explain {} simply.", "What is interesting about {}?"]
    prompts = [f.format(t) for t in topics for f in forms]
    texts = []
    gen_model = model
    tok.padding_side = "left"
    for i in range(0, len(prompts), 40):
        batch = [chat_ids(tok, p, LORA_STYLE) for p in prompts[i:i + 40]]
        enc = tok(batch, return_tensors="pt", padding=True).to(DEV)
        with torch.no_grad():
            out = gen_model.generate(**enc, max_new_tokens=120, do_sample=True, temperature=0.8, pad_token_id=tok.pad_token_id)
        for p, o in zip(prompts[i:i + 40], out):
            ans = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            texts.append(chat_ids(tok, p, None) + ans + tok.eos_token)
    # wrap
    wrapped = []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear) and cname in LORA_TARGETS:
                lw = LoRALinear(child)
                setattr(mod, cname, lw)
                wrapped.append((f"{name}.{cname}", mod, cname, lw))
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for *_, lw in wrapped:
        lw.A.requires_grad_(True)
        lw.B.requires_grad_(True)
        params += [lw.A, lw.B]
    opt = torch.optim.AdamW(params, lr=3e-4)
    tok.padding_side = "right"
    rng = random.Random(0)
    model.train()
    t0 = time.time()
    for step in range(120):
        batch = rng.sample(texts, 4)
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEV)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        loss = model(**enc, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 30 == 0:
            print(f"  lora[{fam}] step {step} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    deltas = {}
    for full, mod, cname, lw in wrapped:
        deltas[full + ".weight"] = ((lw.B @ lw.A) * lw.scale).detach().cpu()
        setattr(mod, cname, lw.base)
    tok.padding_side = "left"
    torch.save(deltas, cache)
    return deltas


# ------------------------------------------------------------------ generation
def generate(model, tok, texts, decoding, watermark, default_temp, bs=80):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar prompts together: batches end early
    texts = [texts[i] for i in order]
    outs = []
    tok.padding_side = "left"
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(DEV)
        kw = dict(max_new_tokens=256, pad_token_id=tok.pad_token_id)
        if decoding == "greedy":
            kw.update(do_sample=False)
        elif decoding == "beam":
            kw.update(do_sample=False, num_beams=4)
        else:
            kw.update(do_sample=True, temperature=1.0 if decoding == "t1" else default_temp, top_p=1.0, top_k=0)
        if watermark:
            kw["logits_processor"] = LogitsProcessorList([KGW(model.config.vocab_size)])
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        outs += [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]
    res = [None] * len(outs)
    for k, i in enumerate(order):
        res[i] = outs[k]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(HERE / "prompts.jsonl"))
    ap.add_argument("--out", default=str(HERE / "data" / "sim_answers.jsonl"))
    ap.add_argument("--truth", default=None)
    ap.add_argument("--default-temp", type=float, default=0.7)
    ap.add_argument("--repeats-scale", type=int, default=1, help="multiply repeats (to study budget)")
    ap.add_argument("--only", default="AB")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    truth_path = Path(a.truth) if a.truth else out.with_name(out.stem + "_truth.csv")
    prompts = read_jsonl(a.prompts)
    jobs = [(p["id"], r, p["prompt"]) for p in prompts for r in range(p["repeats"] * a.repeats_scale)]

    specs = [(fam, v) for fam in a.only for v in VARIANTS[fam]]
    order = list(range(len(specs)))
    random.Random(a.seed).shuffle(order)                     # hide the kind behind a random endpoint id
    names = {i: f"sim-{k+1:02d}" for k, i in enumerate(order)}
    done = {(r["endpoint"], r["id"], r["repeat"]) for r in read_jsonl(out)}
    truth_rows = []
    fh = out.open("a", encoding="utf-8")
    for fam in a.only:
        tok = AutoTokenizer.from_pretrained(FAMILIES[fam])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(FAMILIES[fam], dtype=DTYPE).to(DEV).eval()
        base_sd = {k: v.detach().clone().cpu() for k, v in base.state_dict().items()}
        deltas = None
        for idx, (f2, (kind, mlabel, weights, system, decoding, wm)) in enumerate(specs):
            if f2 != fam:
                continue
            ep = names[idx]
            truth_rows.append((ep, fam, fam + mlabel, kind))
            todo = [j for j in jobs if (ep, j[0], j[1]) not in done]
            if not todo:
                continue
            t0 = time.time()
            base.load_state_dict(base_sd)
            if weights in ("fp8", "int4"):
                quantize_(base, weights)
            elif isinstance(weights, tuple):
                if deltas is None:
                    deltas = train_lora(base, tok, fam, out.parent / f"lora_{fam}.pt")
                    base.load_state_dict(base_sd)
                sd = base.state_dict()
                with torch.no_grad():
                    for k, d in deltas.items():
                        sd[k].add_(weights[1] * d.to(sd[k].device))
            texts = [chat_ids(tok, pr, system) for _, _, pr in todo]
            resp = generate(base, tok, texts, decoding, wm, a.default_temp)
            for (pid, rep, _), r in zip(todo, resp):
                fh.write(json.dumps({"endpoint": ep, "id": pid, "repeat": rep, "response": r}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{fam}] {ep} {kind:10s} {len(todo)} answers in {time.time()-t0:.0f}s", flush=True)
        del base
        if DEV == "mps":
            torch.mps.empty_cache()
    fh.close()
    with truth_path.open("w") as t:
        t.write("endpoint,family,model,kind\n")
        for row in sorted(truth_rows):
            t.write(",".join(row) + "\n")
    print("->", out, truth_path)


if __name__ == "__main__":
    main()
