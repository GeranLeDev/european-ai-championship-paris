# CONTEXTE AGENT — Model Equality Testing (Gao, Liang, Guestrin — Stanford, ICLR 2025, arXiv 2410.20247)
Source vérifiée : texte complet lu (corps + annexe A.2/B). Package Python + code + dataset (1M complétions, 5 modèles) : https://github.com/i-gao/model-equality-testing

## En une phrase
Pour savoir si une API boîte noire sert vraiment le modèle annoncé (ou une version quantifiée/watermarkée/finetunée), on fait un **test à deux échantillons** basé sur la **MMD avec un noyau de Hamming** entre complétions de l'API (Q) et d'une référence (P). Seulement du texte échantillonné, pas de logprobs.

## Formalisation
- Tâche = distribution π sur m prompts + longueur max L. N paires (prompt, complétion) de P et de Q, mêmes paramètres de décodage (temp 1).
- `H0 : π(x)P(y|x) = π(x)Q(y|x)` vs `H1 : ≠`. Erreur de type 1 contrôlée à α.

## Méthode
- Noyau : `k(z,z') = 1{x = x'} · k̃(y, y')` (on ne compare que les complétions d'un même prompt).
- **Noyau Hamming** (le gagnant) : `k̃(y,y') = Σ_{i=1..L} 1{y_i = y'_i}` (padding à droite avec un token spécial). Pseudo-métrique (ne sépare pas toutes les distributions : ne voit que les marginales par position).
- Statistique (MMD² empirique, sans biais) :
  `MMD = 1/(N(N−1)) [Σ_{z,z'∈DP, z≠z'} k + Σ_{z,z'∈DQ, z≠z'} k] − 2/N² Σ_{z∈DP} Σ_{z'∈DQ} k`
- p-value, 2 options :
  1. **Simuler la nulle** : tirer B fois DP, DQ ~ P, calculer la MMD ; p = fraction ≥ observée (demande beaucoup d'échantillons de P ; utilisé dans le papier, B=1000).
  2. **Permutation** : concaténer DP ∪ DQ, mélanger, re-splitter, recalculer B fois (pas besoin d'échantillons en plus ; tendances identiques).
- Autres tests comparés (moins bons) : MMD all-substrings, MMD one-hot (= test multinomial exact), L1 et χ² deux-échantillons.
- Modèles avec des tokenizers différents → comparer en **espace caractères** (plus dur : |V|≈155k, L≈1000 caractères).

## Setup simulations
- π = 25 prompts de 100 caractères tirés de Wikipédia (en, de, es, fr, ru) : « Continue the paragraph. Do not output anything except the continuation… ». L = 50 tokens, température 1, α = 0.05, 10 distributions de prompts, 100 simulations Monte-Carlo.
- Alternatives Q : quantification nf4, int8, fp16 ; watermark Kirchenbauer (biais 2.5) ; finetune ; autre modèle.
- Modèles : Mistral 7B, Llama-3 8B/70B, Llama-3.1 8B/70B Instruct (+ annexe).

## Résultats clés
- Hamming MMD = meilleure efficacité d'échantillonnage : **77.4 % de puissance médiane à N = 10m (~10 échantillons/prompt, N=250)**. Faible contre fp16 (fp16 ≈ fp32 pour ce noyau). Reste puissant quand L augmente (one-hot/L1/χ² s'effondrent).
- Finetune (Llama-3 8B, lr 1e-6, 1000 exemples) détecté avec > 50 % de puissance dès 1 epoch (42 steps), **même finetuné sur du code hors distribution** → le finetuning modifie le comportement global (implication : unlearning/model editing difficiles à isoler).
- Swaps de modèles (espace caractères) : 100 % de puissance sauf intra-famille Llama (3.1 8B vs 70B : 75 % ; 3 8B vs 3.1 8B : 76 %). Watermark : 0.32, int8 : 0.07, nf4 : 1.00 (P = Llama-3.1 8B).
- La MMD est aussi un **estimateur de distance** : la heatmap de 13 modèles se regroupe par famille (données d'entraînement > taille).

## Audit d'APIs commerciales (été 2024)
- 31 endpoints, 9 fournisseurs (Amazon Bedrock, Anyscale, Azure, Deepinfra, Fireworks, Groq, Perplexity, Replicate, Together), Llama-3/3.1 8B/70B.
- Nulle composite : poids fp32 (P1) OU fp16 (P2) ; rejet si `p1 < α ET p2 < α` (garde le FPR ≤ α).
- Prompts : Wikipedia (m=25, L=50 tokens) + HumanEval & UltraChat (m=20, L=250 tokens / 3000 caractères) pour les 8B. α=0.01, N=10m, 10 répétitions, échec si taux de rejet ≥ 0.5 ; Bonferroni α=0.01/3 sur les 3 tâches.
- **11/31 endpoints flaggés**. Perplexity échoue partout (distribution à plus faible entropie) ; Amazon a plusieurs échecs ; Azure, Deepinfra, Together passent tout. HumanEval/UltraChat font échouer plus d'endpoints que Wikipedia.
- Coût d'un audit < 1 $ (Azure 70B : 0.14 $ Wiki, 0.83 $ HumanEval ; Fireworks 8B : < 0.02 $ pour les 3).
- Certaines déviations (Perplexity Llama-3 8B : 0.03) ≈ distance Llama-3 8B ↔ GPT-3.5 (0.03) → « pire que de servir un autre modèle ».
- MMD vs différence d'accuracy HumanEval : corrélation modérée (R² ≈ 0.39) ; la MMD capte aussi « plusieurs façons d'avoir faux ».

## Limites / angles hackathon
- Le noyau Hamming ne voit pas certaines différences (fp16) ; puissance réduite en espace caractères ; suppose température/paramètres identiques ; nulle composite pessimiste.
- Idées : outil d'audit d'API (dashboard « quel modèle sert cet endpoint ? », p-value + distance) ; fingerprinting de modèle/identification parmi K candidats (via la matrice de distances) ; détection de dérive dans le temps (P = API à t-1) ; détecter quantification ou watermark ; **lien avec l'unlearning** : tester si un modèle « désappris » diffère statistiquement de l'original/retrain ; noyaux plus puissants (embeddings) ; extension à d'autres modalités (images — suggérée par les auteurs).

## Pseudo-code minimal
```python
def k_ham(y, yp, L, pad=-1):  # y: list of tokens/chars
    y, yp = (y+[pad]*L)[:L], (yp+[pad]*L)[:L]
    return sum(a == b for a, b in zip(y, yp))
def mmd(DP, DQ, L):  # D = list of (prompt_id, completion)
    k = lambda z, w: (z[0] == w[0]) * k_ham(z[1], w[1], L)
    N = len(DP)
    pp = sum(k(a, b) for i, a in enumerate(DP) for j, b in enumerate(DP) if i != j)
    qq = sum(k(a, b) for i, a in enumerate(DQ) for j, b in enumerate(DQ) if i != j)
    pq = sum(k(a, b) for a in DP for b in DQ)
    return (pp + qq) / (N * (N - 1)) - 2 * pq / N**2
def perm_pvalue(DP, DQ, L, B=1000):
    obs = mmd(DP, DQ, L); D = DP + DQ; cnt = 0
    for _ in range(B):
        random.shuffle(D); cnt += mmd(D[:len(DP)], D[len(DP):], L) >= obs
    return cnt / B
```
(Optimisation : calculer la matrice de Gram une fois, vectorisée numpy, puis permuter les indices.)
