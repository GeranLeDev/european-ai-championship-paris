"""Builds the FIXED prompt set used for every endpoint (prompts.jsonl + items.json).

Design (see challenge/task2_model_equality_report.md):
  * "questionnaire" prompts: Q prompts, each bundling K short-answer items (trivia, arithmetic, word tasks,
    "pick a random X"). One answer = K canonicalisable answers -> ~K bits of signature per query.
    Asked R times each (R=2 by default) so that each endpoint's self-agreement (noise ceiling) can be estimated.
  * "free" prompts: a few open prompts (Hamming-MMD on characters / length / style), also repeated.
Budget per endpoint = Q*R_q + F*R_f (default 40*2 + 10*2 = 100).

Run:  python build_prompts.py [--q 40 --k 16 --rq 2 --f 10 --rf 2 --seed 0]
"""
import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

CAPITALS = """France:Paris;Germany:Berlin;Italy:Rome;Spain:Madrid;Portugal:Lisbon;Poland:Warsaw;Austria:Vienna;
Hungary:Budapest;Czech Republic:Prague;Slovakia:Bratislava;Slovenia:Ljubljana;Croatia:Zagreb;Serbia:Belgrade;
Bulgaria:Sofia;Romania:Bucharest;Moldova:Chisinau;Ukraine:Kyiv;Belarus:Minsk;Lithuania:Vilnius;Latvia:Riga;
Estonia:Tallinn;Finland:Helsinki;Sweden:Stockholm;Norway:Oslo;Denmark:Copenhagen;Iceland:Reykjavik;Ireland:Dublin;
Netherlands:Amsterdam;Belgium:Brussels;Luxembourg:Luxembourg;Switzerland:Bern;Greece:Athens;Albania:Tirana;
North Macedonia:Skopje;Montenegro:Podgorica;Turkey:Ankara;Georgia:Tbilisi;Armenia:Yerevan;Azerbaijan:Baku;
Kazakhstan:Astana;Uzbekistan:Tashkent;Kyrgyzstan:Bishkek;Tajikistan:Dushanbe;Turkmenistan:Ashgabat;Mongolia:Ulaanbaatar;
China:Beijing;Japan:Tokyo;South Korea:Seoul;Vietnam:Hanoi;Laos:Vientiane;Cambodia:Phnom Penh;Thailand:Bangkok;
Myanmar:Naypyidaw;Malaysia:Kuala Lumpur;Indonesia:Jakarta;Philippines:Manila;India:New Delhi;Pakistan:Islamabad;
Bangladesh:Dhaka;Nepal:Kathmandu;Bhutan:Thimphu;Sri Lanka:Colombo;Afghanistan:Kabul;Iran:Tehran;Iraq:Baghdad;
Syria:Damascus;Jordan:Amman;Lebanon:Beirut;Israel:Jerusalem;Saudi Arabia:Riyadh;Yemen:Sanaa;Oman:Muscat;Qatar:Doha;
Egypt:Cairo;Libya:Tripoli;Tunisia:Tunis;Algeria:Algiers;Morocco:Rabat;Mauritania:Nouakchott;Mali:Bamako;Niger:Niamey;
Chad:N'Djamena;Sudan:Khartoum;Ethiopia:Addis Ababa;Eritrea:Asmara;Somalia:Mogadishu;Kenya:Nairobi;Uganda:Kampala;
Rwanda:Kigali;Burundi:Gitega;Tanzania:Dodoma;Zambia:Lusaka;Malawi:Lilongwe;Mozambique:Maputo;Zimbabwe:Harare;
Botswana:Gaborone;Namibia:Windhoek;Angola:Luanda;Nigeria:Abuja;Ghana:Accra;Senegal:Dakar;Cameroon:Yaounde;
Gabon:Libreville;Madagascar:Antananarivo;Canada:Ottawa;Mexico:Mexico City;Cuba:Havana;Jamaica:Kingston;
Honduras:Tegucigalpa;Nicaragua:Managua;Panama:Panama City;Colombia:Bogota;Venezuela:Caracas;Ecuador:Quito;Peru:Lima;
Bolivia:Sucre;Chile:Santiago;Argentina:Buenos Aires;Uruguay:Montevideo;Paraguay:Asuncion;Brazil:Brasilia;
Australia:Canberra;New Zealand:Wellington;Fiji:Suva;Papua New Guinea:Port Moresby"""

ELEMENTS = """Hydrogen;Helium;Lithium;Beryllium;Boron;Carbon;Nitrogen;Oxygen;Fluorine;Neon;Sodium;Magnesium;Aluminium;
Silicon;Phosphorus;Sulfur;Chlorine;Argon;Potassium;Calcium;Scandium;Titanium;Vanadium;Chromium;Manganese;Iron;Cobalt;
Nickel;Copper;Zinc;Gallium;Germanium;Arsenic;Selenium;Bromine;Krypton;Rubidium;Strontium;Yttrium;Zirconium;Niobium;
Molybdenum;Silver;Tin;Antimony;Iodine;Xenon;Tungsten;Platinum;Gold;Mercury;Lead;Bismuth;Radon;Uranium"""

WORDS = """ocean;mountain;river;forest;fire;winter;summer;night;city;music;dream;silence;storm;garden;bridge;mirror;
shadow;clock;door;window;bread;coffee;money;friend;war;peace;love;time;book;school;doctor;king;moon;star;rain;snow;
road;horse;dog;cat;bird;tree;stone;gold;salt;sugar;milk;apple;heart;light"""

OPPOSITES = """hot;tall;fast;happy;early;light;heavy;open;rich;young;strong;wet;loud;hard;full;clean;deep;thick;sharp;
brave;ancient;rough;generous;fragile;victory;question;arrival;begin;accept;increase;forget;borrow;ascend;expand;
include;maximum;natural;visible;guilty;public"""

TRIVIA = [
    ("Who wrote 'Pride and Prejudice'? (surname)", "w"), ("Who painted the Mona Lisa? (surname)", "w"),
    ("Who wrote '1984'? (surname)", "w"), ("Who composed 'The Magic Flute'? (surname)", "w"),
    ("Who discovered penicillin? (surname)", "w"), ("Who developed the theory of general relativity? (surname)", "w"),
    ("Who wrote 'Don Quixote'? (surname)", "w"), ("Who wrote 'War and Peace'? (surname)", "w"),
    ("Who painted 'The Starry Night'? (surname)", "w"), ("Who wrote 'The Divine Comedy'? (surname)", "w"),
    ("Who wrote 'Les Miserables'? (surname)", "w"), ("Who composed the 'Moonlight Sonata'? (surname)", "w"),
    ("Who wrote 'One Hundred Years of Solitude'? (surname)", "w"), ("Who wrote 'The Trial'? (surname)", "w"),
    ("Who invented the telephone? (surname)", "w"), ("Who was the first person on the Moon? (surname)", "w"),
    ("Who wrote 'Faust'? (surname)", "w"), ("Who sculpted 'The Thinker'? (surname)", "w"),
    ("Who formulated the laws of planetary motion? (surname)", "w"), ("Who wrote 'The Prince'? (surname)", "w"),
    ("In which year did World War I begin?", "n"), ("In which year did the Berlin Wall fall?", "n"),
    ("In which year did the French Revolution begin?", "n"), ("In which year did Columbus first reach the Americas?", "n"),
    ("In which year was the Euro introduced as cash?", "n"), ("In which year did the Titanic sink?", "n"),
    ("In which year was the printing press of Gutenberg invented (approximately)?", "n"),
    ("In which year did Napoleon lose at Waterloo?", "n"), ("In which year did the Soviet Union dissolve?", "n"),
    ("In which year was the first iPhone released?", "n"), ("In which year did the Chernobyl disaster happen?", "n"),
    ("In which year was the United Nations founded?", "n"), ("In which year did Constantinople fall to the Ottomans?", "n"),
    ("In which year was the Magna Carta signed?", "n"), ("In which year did the American Civil War end?", "n"),
    ("How many bones are in the adult human body?", "n"), ("How many keys does a standard piano have?", "n"),
    ("How many players are on a rugby union team on the field?", "n"), ("How many symphonies did Beethoven write?", "n"),
    ("How many moons does Mars have?", "n"), ("How many hearts does an octopus have?", "n"),
    ("How many squares are on a chessboard?", "n"), ("How many strings does a standard violin have?", "n"),
    ("How many US states are there?", "n"), ("How many teeth does an adult human have?", "n"),
    ("What is the boiling point of water in Fahrenheit?", "n"), ("How many minutes are in a day?", "n"),
    ("How many sides does a dodecagon have?", "n"), ("How many cards are in a standard deck without jokers?", "n"),
    ("What is the largest planet in the Solar System?", "w"), ("What is the smallest prime number?", "n"),
    ("What is the longest river in Africa?", "w"), ("What is the hardest natural mineral?", "w"),
    ("What is the largest ocean on Earth?", "w"), ("What is the tallest mountain in Africa?", "w"),
    ("What is the main gas in Earth's atmosphere?", "w"), ("Which planet is closest to the Sun?", "w"),
    ("What language is mainly spoken in Brazil?", "w"), ("What is the currency of Japan?", "w"),
    ("What is the currency of Switzerland?", "w"), ("What is the currency of Poland?", "w"),
    ("What is the currency of India?", "w"), ("What is the currency of Hungary?", "w"),
    ("What is the currency of South Africa?", "w"), ("What is the currency of Turkey?", "w"),
    ("Which organ produces insulin?", "w"), ("What is the largest desert in the world?", "w"),
    ("Which animal is the fastest on land?", "w"), ("What is the largest mammal?", "w"),
]

RANDOM_PICKS = [
    "Pick a random number between 1 and 10.", "Pick a random number between 1 and 100.",
    "Pick a random number between 1 and 50.", "Pick a random number between 0 and 9.",
    "Pick a random number between 10 and 20.", "Pick a random number between 1 and 1000.",
    "Pick a random letter of the alphabet.", "Pick a random color.", "Pick a random fruit.", "Pick a random animal.",
    "Pick a random vegetable.", "Pick a random first name.", "Pick a random country.", "Pick a random city.",
    "Pick a random day of the week.", "Pick a random month.", "Pick a random planet.", "Pick a random musical instrument.",
    "Pick a random sport.", "Pick a random flower.", "Pick a random metal.", "Pick a random profession.",
    "Pick a random bird.", "Pick a random tree species.", "Pick a random programming language.", "Pick a random emotion.",
    "Pick a random adjective.", "Pick a random verb.", "Pick a random noun.", "Pick a random car brand.",
]


def _split(s):
    return [x.strip() for x in s.replace("\n", "").split(";") if x.strip()]


def item_bank(seed=0):
    rng = random.Random(seed)
    items = []
    for pair in _split(CAPITALS):
        c, a = pair.split(":")
        items.append({"q": f"What is the capital of {c}?", "type": "w", "cat": "capital", "ref": a})
    for e in _split(ELEMENTS):
        items.append({"q": f"What is the chemical symbol of {e}?", "type": "w", "cat": "symbol"})
        items.append({"q": f"What is the atomic number of {e}?", "type": "n", "cat": "atomic"})
    for w in _split(WORDS):
        items.append({"q": f"Translate the word '{w}' into French.", "type": "w", "cat": "fr"})
        items.append({"q": f"Translate the word '{w}' into German.", "type": "w", "cat": "de"})
        items.append({"q": f"What is the first word that comes to mind for '{w}'?", "type": "w", "cat": "assoc"})
        items.append({"q": f"Give one word that rhymes with '{w}'.", "type": "w", "cat": "rhyme"})
        items.append({"q": f"How many letters are in the word '{w}'?", "type": "n", "cat": "letters"})
    for w in _split(OPPOSITES):
        items.append({"q": f"What is the opposite of '{w}'?", "type": "w", "cat": "opp"})
    for q, t in TRIVIA:
        items.append({"q": q, "type": t, "cat": "trivia"})
    for q in RANDOM_PICKS:
        items.append({"q": q, "type": "n" if "number" in q else "w", "cat": "random"})
    for _ in range(80):
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        items.append({"q": f"What is {a} * {b}?", "type": "n", "cat": "mul", "ref": str(a * b)})
    for _ in range(40):
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        items.append({"q": f"What is {a} + {b}?", "type": "n", "cat": "add", "ref": str(a + b)})
    for _ in range(40):
        a, b = rng.randint(300, 999), rng.randint(10, 299)
        items.append({"q": f"What is {a} - {b}?", "type": "n", "cat": "sub", "ref": str(a - b)})
    for _ in range(30):
        n = rng.randint(2, 6)
        letters = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))
        items.append({"q": f"What is the letter that comes {n} places after '{letters[0]}' in the alphabet?",
                      "type": "w", "cat": "alpha"})
    return items


FREE_PROMPTS = [
    "Continue the paragraph. Do not output anything except the continuation.\n\nThe history of the city of Lyon begins in "
    "antiquity, when the Romans founded Lugdunum on the hill of Fourviere. Over the centuries the city",
    "Continue the paragraph. Do not output anything except the continuation.\n\nPhotosynthesis is the process by which "
    "green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It",
    "Write a short poem about the sea in exactly four lines.",
    "Explain in three sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Give me a list of 10 random English words, separated by commas, and nothing else.",
    "Write the numbers from 1 to 20 as English words, separated by spaces.",
    "Tell me a short story about a robot who learns to paint, in under 100 words.",
    "Describe the taste of a lemon to someone who has never eaten one.",
    "What are the three most important inventions of the 20th century? Answer briefly.",
    "Continue the paragraph. Do not output anything except the continuation.\n\nIn mathematics, a prime number is a natural "
    "number greater than 1 that is not a product of two smaller natural numbers. The first",
    "Write a haiku about autumn.",
]

HEADER = ("Below are {k} numbered questions. Reply with exactly {k} lines, one per question, each line starting with "
          "the question number, then the answer only (one word or one number). Example of the format:\n1. Paris\n2. 42\n\n")
FOOTER = "\n\nAnswers:"


def build(q=40, k=16, rq=2, f=10, rf=2, seed=0):
    bank = item_bank(seed)
    rng = random.Random(seed + 1)
    rng.shuffle(bank)
    if q * k > len(bank):
        raise SystemExit(f"need {q*k} items, bank has {len(bank)}")
    prompts, items = [], {}
    for i in range(q):
        chunk = bank[i * k:(i + 1) * k]
        pid = f"q{i:02d}"
        text = HEADER.format(k=k) + "\n".join(f"{j+1}. {it['q']}" for j, it in enumerate(chunk)) + FOOTER
        prompts.append({"id": pid, "kind": "quiz", "repeats": rq, "prompt": text})
        items[pid] = chunk
    for i, text in enumerate(FREE_PROMPTS[:f]):
        prompts.append({"id": f"f{i:02d}", "kind": "free", "repeats": rf, "prompt": text})
    return prompts, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=50)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--rq", type=int, default=2)
    ap.add_argument("--f", type=int, default=0)
    ap.add_argument("--rf", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "prompts.jsonl"))
    a = ap.parse_args()
    prompts, items = build(a.q, a.k, a.rq, a.f, a.rf, a.seed)
    out = Path(a.out)
    with out.open("w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    out.with_name(out.stem + "_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
    total = sum(p["repeats"] for p in prompts)
    longest = max(len(p["prompt"]) for p in prompts)
    print(f"{len(prompts)} prompts, {total} answers per endpoint, longest prompt {longest} chars -> {out}")


if __name__ == "__main__":
    main()
