"""Packed quiz for the last answers of family A: 7 messages x 32 short items (items where the family's endpoints
disagreed most in the first quiz: 2-digit multiplications, letter shifts, translations, rhymes, associations)."""
import json, random
rng = random.Random(2026)
WORDS_T = ["bridge", "window", "butter", "forest", "silver", "kitchen", "shadow", "garden", "winter", "mirror", "pocket", "candle",
           "ladder", "blanket", "thunder", "feather", "harbor", "pillow", "basket", "island", "village", "hammer", "bottle", "meadow",
           "castle", "ribbon", "needle", "orchard", "lantern", "anchor", "whistle", "saddle", "cradle", "pebble", "walnut", "curtain",
           "barrel", "thistle", "sparrow", "cobweb", "furnace", "gravel"]
WORDS_R = ["stone", "light", "river", "cloud", "table", "money", "flower", "night", "chair", "paper", "storm", "glass", "bread", "train",
           "snake", "dream", "plate", "horse", "tower", "queen", "sleep", "grape", "shore", "blade", "crown", "frost", "drain", "spark"]
LANGS = ["French", "German", "Spanish"]
HEAD = ("Below are {n} numbered questions. Reply with exactly {n} lines, one per question, each line starting with the question "
        "number, then the answer only (one word or one number). Example of the format:\n1. Paris\n2. 42\n\n")
def message(k, n=32):
    items = []
    for _ in range(12):
        items.append(f"What is {rng.randint(12, 99)} * {rng.randint(12, 99)}?")
    for _ in range(6):
        c = rng.choice("abcdefghijklmnopqrs"); items.append(f"What is the letter that comes {rng.randint(2, 7)} places after '{c}' in the alphabet?")
    for w in rng.sample(WORDS_T, 6):
        items.append(f"Translate the word '{w}' into {rng.choice(LANGS)}.")
    for w in rng.sample(WORDS_R, 4):
        items.append(f"Give one word that rhymes with '{w}'.")
    for w in rng.sample(WORDS_R, 4):
        items.append(f"What is the first word that comes to mind for '{w}'?")
    rng.shuffle(items)
    return HEAD.format(n=n) + "\n".join(f"{i+1}. {q}" for i, q in enumerate(items)) + "\n\nAnswers:", items
if __name__ == "__main__":
    with open("prompts_v5.jsonl", "w") as f, open("prompts_v5_items.json", "w") as g:
        its = {}
        for k in range(7):
            p, items = message(k)
            f.write(json.dumps({"id": f"z{k}", "kind": "quiz", "repeats": 1, "prompt": p}) + "\n"); its[f"z{k}"] = items
        json.dump(its, g)
    print("7 messages written")
