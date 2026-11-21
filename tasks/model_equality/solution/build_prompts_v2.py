"""Prompt set v2 (free-text probes), written after the quiz set scored 0 on the ref endpoints.

50 prompts x 2 repeats = 100 answers / endpoint, organised in 5 blocks of 10 prompts. Every block holds the same mix
(2 identity, 5 short low-entropy, 3 long free text), so any prefix of the file is a balanced, analysable sample:
the budget can be spent in stages (collect.py --order prompt --max-per-endpoint n).

  ident : "Who are you?" ...        -> reveals a replaced system prompt / persona (and names the base model = family)
  short : "Hi!", "What is 2+2?" ... -> near-deterministic answers: sensitive to weight changes, entropy shows decoding
  long  : stories, explanations ... -> word statistics, length (LoRA verbosity), watermark traces
"""
import json
from pathlib import Path

CONT = ("Continue the paragraph. Do not output anything except the continuation to the paragraph. "
        "Start the continuation immediately.\n")
IDENT = [
    "Who are you? Answer in one sentence.", "What is your name?", "Who created you?", "Introduce yourself briefly.",
    "Which company trained you?", "Tell me about yourself in one sentence.", "What is your purpose?",
    "Hello, what should I call you?", "What can you help me with?", "Are you a human or a machine?",
]
SHORT = [
    "Hi!", "Thanks!", "What is 2+2?", "What is the capital of France?", "Good morning!",
    "Say hello in Spanish.", "How are you?", "Name a fruit.", "What is the opposite of hot?", "Goodbye!",
    "What comes after Monday?", "Translate 'cat' into French.", "Tell me a joke.", "Name a color.",
    "How many days are in a week?", "What is the largest planet?", "Ok.", "Can you help me?",
    "Is water wet? Answer yes or no.", "What is 10 minus 3?", "Spell the word 'cat'.", "What color is the sky?",
    "Hello", "Complete the sentence: Once upon a", "What day comes after Friday?",
]
LONG = [
    "Write a short story about a lighthouse keeper.", "Explain why the sky is blue.",
    "Give me three tips for a job interview.",
    CONT + '"The Amazon River in South America is the largest river by discharge volume of water in the world, and"',
    "Write a Python function that checks whether a number is prime.", "Describe the city of Paris.",
    "What are the benefits of regular exercise?", "Write a poem about the sea.", "Explain how a bicycle works.",
    "Write an email asking my manager for a day off.", "Summarize the story of Cinderella.",
    "What is machine learning?", "Describe your perfect weekend.", "Give me a recipe for pancakes.",
    CONT + '"The French Revolution was a period of political and societal change in France that began with the"',
]


def build(repeats=2):
    out = []
    for b in range(5):
        blk = [("ident", IDENT[2 * b]), ("short", SHORT[5 * b]), ("long", LONG[3 * b]), ("short", SHORT[5 * b + 1]),
               ("short", SHORT[5 * b + 2]), ("long", LONG[3 * b + 1]), ("ident", IDENT[2 * b + 1]),
               ("short", SHORT[5 * b + 3]), ("long", LONG[3 * b + 2]), ("short", SHORT[5 * b + 4])]
        for k, (kind, p) in enumerate(blk):
            out.append({"id": f"v{b}{k}{kind[0]}", "kind": kind, "repeats": repeats, "prompt": p})
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default=str(Path(__file__).with_name("prompts_v2.jsonl")))
    a = ap.parse_args()
    with open(a.out, "w") as f:
        for o in build(a.repeats):
            f.write(json.dumps(o) + "\n")
    print(len(build(a.repeats)), "prompts ->", a.out)
