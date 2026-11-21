# Analyse technique : Model Equality Testing (arXiv 2410.20247, ICLR 2025)

Auteurs : Irena Gao, Percy Liang, Carlos Guestrin (Stanford University).
Source : `papers/txt/model_equality_testing.txt` (version arXiv v2, 8 avril 2025).

Convention : tout ce qui n'est pas explicitement dans le papier est marqué **(inférence)**.

---

## TL;DR

1. Auditer une API LLM revient à un **test à deux échantillons** : on compare des complétions d'une référence P et de l'API Q sur les mêmes prompts, avec H0 : P = Q.
2. Le test **MMD avec noyau de Hamming** (string kernel simple) atteint une **puissance médiane de 77,4 %** sur quantification, watermark et finetuning, avec en moyenne **~10 échantillons par prompt**.
3. Appliqué à **31 endpoints Llama** de 9 fournisseurs (été 2024), le test rejette **11 endpoints** ; chaque audit coûte **moins de 1 $**.

---

## 1. Problème et formalisation

### Contexte
- Les fournisseurs d'API (Bedrock, Azure AI Studio, Together, Fireworks, etc.) peuvent **quantifier**, **watermarker**, **finetuner** le modèle, ou mal implémenter les paramètres de décodage. Tout cela modifie la distribution des complétions, sans forcément être annoncé.
- Deux questions : (1) comment l'API a-t-elle modifié la distribution du modèle ? (2) l'API change-t-elle dans le temps ? (§1)
- Les audits existants (ArtificialAnalysis, etc.) utilisent des benchmarks à réponse courte, souvent en décodage glouton. Ils ne vérifient que les modes des distributions de tokens, pas la distribution sur les complétions (§1).

### Cadre formel
- Tâche : distribution **π** sur **m** prompts x, longueur maximale de complétion **L** tokens. Vocabulaire commun **V**, mêmes paramètres de décodage pour P et Q.
- Échantillons (eq. 1) :
  - D_P = { z(i) : x(i) ~ π, y(i) ~ P(· | x(i)) }, i = 1..N
  - D_Q = { z(i) : x(i) ~ π, y(i) ~ Q(· | x(i)) }, i = 1..N
  - Avec z = (x, y) : une paire prompt-complétion.
- Hypothèses (eq. 2) :
  - H0 : π(x) P(y|x) = π(x) Q(y|x)
  - H1 : π(x) P(y|x) ≠ π(x) Q(y|x)
- Contrainte : le **taux d'erreur de type I est contrôlé à α**. Objectif : maximiser la **puissance** contre Q inconnu, et généraliser à plusieurs modèles et distributions de prompts.
- Le papier vise des tests **sample-efficient** : puissants quand N est petit devant |V| et L. Exemple : Llama-3 a |V| = 128 256 et les usages longs vont à L = 250 (note 4).
- Nombre d'éléments de l'espace prompt-complétion : m·|V|^L (§2). Les distributions réelles sont supposées de faible dimension, car peu de tokens portent la masse à chaque position (§2).

### Ce que le test ne donne pas
- Il dit si P ≠ Q, pas *ce qui* a changé. Le papier recommande un examen manuel quand l'effet estimé (MMD) est grand (App. C.8).

---

## 2. Méthode en détail

### 2.1 MMD (Maximum Mean Discrepancy)

Pour un noyau k de feature map φ, unitaire en norme (§3, eq. 3) :

```
MMD_k(πP, πQ) = || E_{z~πP}[φ(z)] − E_{z~πQ}[φ(z)] ||²
              = E_{z,z'~πP}[k(z,z')] + E_{z,z'~πQ}[k(z,z')] − 2·E_{z~πP, z'~πQ}[k(z,z')]
```

Noyaux de la forme **k(z, z') = 1{x = x'} · k̃(y, y')**, où k̃ est un noyau sur les complétions indépendant du prompt. Alors :

```
MMD_k(πP, πQ) = E_{x~π} [ MMD_k̃( P(·|x), Q(·|x) ) ]
```

Le MMD est donc une moyenne, sur les prompts, du MMD entre distributions conditionnelles.

### 2.2 Estimateur (eq. 4)

Estimateur empirique utilisé comme statistique de test :

```
MMD^(D_P, D_Q) = 1/(N(N−1)) · [ Σ_{z,z'∈D_P} k(z,z') + Σ_{z,z'∈D_Q} k(z,z') ]
               − 2/N² · Σ_{z∈D_P} Σ_{z'∈D_Q} k(z,z')
```

(inférence) : les sommes "intra" sont lues comme des paires distinctes (i ≠ j), ce qui donne l'estimateur sans biais. Le papier n'écrit pas explicitement "i ≠ j". Comme k = 1{x=x'}·k̃, seules les paires de même prompt contribuent.

### 2.3 Noyaux testés (§3, App. A.1)

| Noyau | Formule | Propriétés (papier) |
|---|---|---|
| **All-substrings** | k̃_all(y,y') = Σ_{s ∈ V^{≤L}} n_s{y} · n_s{y'} (eq. 5), n_s = nombre d'occurrences de s | Universel, MMD métrique. Très coûteux à calculer. |
| **One-hot** | k̃(y,y') = 1{y = y'} (eq. 6) | Universel, MMD métrique. Revient à un test multinomial classique. Peu puissant en petit échantillon : il faut des collisions exactes entre complétions. |
| **Hamming** | k̃(y,y') = Σ_{i=1}^{L} 1{y_i = y'_i} (eq. 7) | **Pseudo-métrique** (non injective, App. A.1). Rapide. Plus puissant en petit N. y plus court que L est complété par un token spécial. |
| Levenshtein (App. C.1) | k(x,y) = exp(−Levenshtein(x,y)) | Moins puissant que Hamming, mais toujours meilleur que les tests classiques (Table 9). |

Argument du papier pour Hamming : quantification, watermark, finetuning produisent des **distances de Hamming inter-échantillons détectables** (§3).

Preuve de non-injectivité (App. A.1) : l'embedding moyen du noyau de Hamming empile les marginales P(x_i = v), qui ne déterminent pas la loi jointe. Donc deux distributions P ≠ Q peuvent avoir le même MMD nul.

### 2.4 Autres tests comparés (§4.1)

- **L1 multinomial** (eq. 8) : Σ_z | n_z{D_P}/N − n_z{D_Q}/N |
- **χ² à deux échantillons** (eq. 9) : formule tronquée de Balakrishnan & Wasserman (2018) et Bhattacharya & Valiant (2015).

Tests de **goodness-of-fit** (App. C.5, nécessitent P(y|x)) : L1gof, L2gof, χ² de Pearson, χ² tronqué, rapport de vraisemblance (LRT). Résultat : les meilleurs (χ² tronqué, L2) battent les tests à deux échantillons seulement pour N < 1000. Au-delà, l'avantage s'inverse, car l'approximation de P introduit un biais.

### 2.5 Calcul de la p-value

Deux méthodes (App. A.2) :

**Algorithme 1 : simulation sous H0.**
Répéter B fois : tirer D_Q^(i) et D_P^(i) **depuis P**, calculer MMD^. La p-value est la proportion des valeurs simulées ≥ à la valeur observée. Les expériences principales utilisent B = 1000 et réutilisent cette distribution nulle pour toutes les alternatives à une taille N donnée. Demande un accès d'échantillonnage important à P.

**Algorithme 2 : permutation (Lehmann et al., 1986).**
Concaténer D_Q et D_P, mélanger, re-découper en deux groupes, recalculer MMD^. Répéter B fois, p-value = proportion ≥ observé. N'exige pas de tirer depuis P, mais la puissance est potentiellement plus faible. Utilisé en App. C.7 avec B = 100, 20 simulations de puissance, seulement Mistral 7B et Llama-3 8B. Résultats : tendances identiques.

Pseudo-code de l'algorithme 2 : voir section 8.

### 2.6 Nombre d'échantillons

- Taille de référence : **N = 10·m** (10 complétions par prompt). Exemple : m = 25 prompts, donc N = 250.
- Les courbes de sample complexity vont de N = 10m à N = 100m.
- Plus de prompts augmente la puissance, à N = 10m (Fig. 8 : m de 1 à 100), à condition d'augmenter N proportionnellement.
- Les échantillons P et Q peuvent être déséquilibrés : augmenter |D_P| avec |D_Q| = 10m donne de légers gains de puissance, surtout pour L1 (App. C.6).

### 2.7 Composite (§5, eq. 10-11)

Quand on ne sait pas si la référence est fp32 ou fp16 :
- H0 : P1 = Q **OU** P2 = Q (P1 = fp32, P2 = fp16).
- H1 : P1 ≠ Q **ET** P2 ≠ Q.
- Règle de rejet (eq. 11) : δ = 1{p1 < α ET p2 < α}.
- Contrôle du FPR à α sous l'H0 composite (preuve dans le papier). Règle "possiblement pessimiste" : la puissance baisse (Table 18).

---

## 3. Setup expérimental

### 3.1 Modèles
- **Simulations (§4)** : Mistral 7B Instruct, Llama-3 8B Instruct, Llama-3 70B Instruct, Llama-3.1 8B Instruct, Llama-3.1 70B Instruct. Additionnels en App. C.1 et C.2 : Phi-3 mini, OLMo 7B, Gemma-2 9B.
- **Model swap (§4.3)** : pool de 13 modèles instruction-tuned, dont 8 open-weight (Llama-3, Llama-3.1, Mistral 7B, Phi-3 mini, OLMo 7B, Gemma-2 9B, etc.) et 5 modèles OpenAI (gpt-4o-mini, gpt-4o, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-4-0125-preview).
- **Audit (§5)** : Llama-3 8B, Llama-3.1 8B, Llama-3 70B, Llama-3.1 70B. Llama-3.1 405B n'est évalué que par distances entre APIs (App. C.9, Fig. 26).

### 3.2 Distorsions
- **Quantification** : nf4 (QLoRA, Dettmers et al. 2024), int8 (LLM.int8, Dettmers et al. 2022), fp16. Implémentations de la bibliothèque Transformers.
- **Watermark** : Kirchenbauer et al. (2023), biais 2.5, context width 1 (valeurs par défaut Transformers).
- **Finetuning (§4.2)** : Llama-3 8B Instruct, learning rate 1e-6, AdamW. Deux jeux de données de 1000 exemples :
  - Wikipedia (i.i.d. avec la tâche de test) ;
  - CodeAlpaca (hors distribution ; Chaudhary 2023).
  - Une époque = 42 étapes d'optimisation.
- **Décodage (App. C.3)** : température τ = 0,5 et τ = 1,5 contre τ = 1.
- **Changement de modèle** : paires entre les 13 modèles (§4.3).

### 3.3 Paramètres de sampling
- Température **1**, sans top-k ni top-p (App. B.1).
- L = **50 tokens** pour Wikipedia, **250 tokens** pour HumanEval et UltraChat.
- En espace caractères (§4.3 et §5) : L ≈ **1000 caractères** (Wikipedia) ou **3000 caractères** (HumanEval/UltraChat). |V| = 155 063.
- Chat template : template par défaut de Transformers pour les modèles. Pour Llama-3.1, le template par défaut inclut la date du jour et ne correspondait pas au nombre de tokens renvoyé par les APIs : les auteurs ont utilisé le template Llama-3 (App. B.1).

### 3.4 Jeux de prompts
- **Wikipedia** (Box 1) : "Continue the paragraph..." appliqué à 100 caractères d'articles Wikipedia (anglais, allemand, espagnol, français, russe ; source Cohere 2023). m = 25 prompts échantillonnés par distribution ; 10 distributions différentes pour généraliser.
- **HumanEval** (Box 3, Chen et al. 2021) : complétion de code, m = 20, avec vérificateur automatique.
- **UltraChat** (Box 4, Ding et al. 2023) : instructions, m = 20, tâche créative.
- Dataset d'échantillons : 440 prompts (80 Wikipedia par langue × 5 = 400, + 20 HumanEval + 20 UltraChat). Environ 1,1 M de complétions (App. B.1). L'intro dit "1 million".

### 3.5 Paramètres de test
- α = 0,05 pour les simulations (§4), α = 0,01 pour les audits (§5).
- Puissance : moyenne du taux de rejet sur **100 simulations** (fresh D_P et D_Q à chaque fois). Exception : all-substrings, 20 simulations, p-values avec 100 échantillons nuls.
- Audit : **N = 10m** ; test répété 10 fois ; endpoint **échoue si le taux moyen de rejet ≥ 0,5**.
- Pour l'audit des 8B : 3 distributions de prompts, combinées par **Bonferroni** (α = 0,01/3, Fig. 1).

### 3.6 Échantillons locaux
- fp32 : 15 000 complétions par prompt ; autres précisions et watermark : 5 000.
- GPU : RTX 6000, RTX 3090, Quadro RTX 8000, A100 ; Transformers et vLLM (App. B.1).

### 3.7 Échantillons API
- Un completion par requête, en boucle, sur la même fenêtre de 24 h quand c'est possible. Limites de débit et erreurs ont étalé certains prompts sur plusieurs jours.
- Dates de collecte : Tables 2 à 7 (juillet à août 2024).
- Paquets Python du fournisseur si disponibles, sinon HTTP brut.

---

## 4. Résultats chiffrés clés

### 4.1 Quantification et watermark (§4.1, Fig. 2)
- **Sample complexity** (Fig. 2 gauche), moyennée sur modèles et distributions, L = 50 :
  - Hamming MMD : **puissance médiane 77,4 %** à N = 10m (~10 échantillons par prompt).
  - Les autres tests sont plus faibles à N équivalent.
- **Robustesse à L** (Fig. 2 milieu, N = 250) : Hamming et all-substrings restent robustes quand L augmente de 1 à 50. Les tests L1 et χ² chutent fortement au-delà de L ≈ 5 tokens.
- **Par distorsion** (Fig. 2 droite) : Hamming fort sur tout sauf **fp16**, où la puissance à N = 10m est faible. Le papier interprète : fp16 et fp32 diffèrent d'une façon que le noyau de Hamming ne capture pas.
- **Table 8** (L = 50, N = 10m, FPR à fp32 ≈ 0,04–0,06) :

| Modèle | watermark | nf4 | int8 |
|---|---|---|---|
| Llama-3 8B | 1.00 | 0.93 | 0.79 |
| Llama-3.1 8B | 0.62 | 1.00 | 0.08 |
| Llama-3 70B | 0.97 | 1.00 | 1.00 |
| Llama-3.1 70B | 0.54 | 1.00 | 1.00 |
| Mistral 7B | 0.72 | 0.89 | 0.30 |
| OLMo 7B | 0.47 | 0.99 | 0.36 |
| Gemma-2 9B | 0.43 | 0.12 | 0.06 |
| Phi-3 mini | 0.75 | 0.63 | 0.24 |

### 4.2 Finetuning (§4.2, Fig. 3 haut gauche)
- Hamming MMD détecte le finetuning de Llama-3 8B avec **puissance > 50 %** dès **une époque** (42 étapes), sur 1000 exemples.
- Plus détectable quand la distribution de finetuning est i.i.d. avec la tâche de test (Wikipedia). Mais le finetuning sur CodeAlpaca (hors distribution) est **aussi détectable** sur les prompts Wikipedia (le papier note que l'effet n'est pas isolé à une seule distribution).

### 4.3 Changement de modèle (§4.3, Fig. 3 bas gauche, Tables 10 et 11)
- En espace caractères (L ≈ 1000, N = 10m) :
  - Tous les changements de modèle sont détectés à **100 %**, sauf les paires **intra-famille Llama** : Llama-3.1 8B vs 70B = **75 %** ; Llama-3 8B vs Llama-3.1 8B = **76 %**.
  - Watermark (0,32 sur Llama-3.1 8B) et int8 (0,07) sont beaucoup moins détectables que les swaps de modèles.
- Dans Fig. 3 bas gauche, la référence est Llama-3.1 8B et les alternatives (lignes) incluent Llama-3 8B 0,76, Llama-3 70B 1,00, Llama-3.1 70B 0,75, OLMo 7B 1,00, gpt-4o-mini 1,00, nf4 1,00, watermark 0,32, int8 0,07.
- **Table 10** (espace caractères, L = 1000, N = 10m) : la puissance baisse par rapport à l'espace tokens. Table 11 : avec N = 50m, watermark passe de 0,62 à 1,00 et nf4 de 0,38 à 1,00 (Llama-3 8B).

### 4.4 Distances entre modèles (§4.3, Fig. 3 droite)
- Le MMD est un estimateur de distance. Calculé sur N = 100m, moyenné sur 10 simulations.
- Les modèles d'une même **famille** se regroupent (données d'entraînement plus importantes que la taille, selon les auteurs).
- Exemples de valeurs : Llama-3 8B vs GPT-3.5 Turbo = 0,03 ; vs Phi-3 mini = 0,04 ; vs OLMo 7B = 0,04.
- Les générations Llama 3 et 3.1 sont proches, mais certaines générations GPT (4-preview et 3.5-turbo) ne le sont pas.

### 4.5 Décodage et distribution des prompts
- **Température** (Table 12) : τ = 0,5 et τ = 1,5 vs τ = 1 : puissance **1,00** pour Llama-3 8B et 3.1 8B, **0,97** pour Mistral (la table écrit "Mistral 8B", coquille probable).
- **Distribution des prompts** (Table 13, token space) :
  - HumanEval : watermark 0,99 ; nf4 1,00 ; int8 0,32.
  - UltraChat : watermark 0,89 ; nf4 1,00 ; int8 0,04.
  - Wikipedia "All languages" : watermark 1,00 ; nf4 0,93 ; int8 0,79.
  - Le papier conclut que la distribution affecte surtout la détection de la quantification, pas celle du watermark. Les tâches créatives ne sont pas plus difficiles que les tâches contraintes.
- **Hamming vs one-hot** (Table 14) : Hamming dépasse one-hot sur toutes les distributions.
- **Levenshtein** (Table 9) : moins puissant que Hamming.

### 4.6 Audit des APIs (§5, Table 1, Table 16)

Nombre d'endpoints : **31** (9 fournisseurs × 4 modèles).

**Table 16 (résultat global, Bonferroni, repris Fig. 1) : 11 endpoints sur 31 échouent.**

| Fournisseur | 3 8B | 3.1 8B | 3 70B | 3.1 70B |
|---|---|---|---|---|
| Amazon Bedrock | ✗ | ✗ | ✗ | ✗ |
| Anyscale | ✓ | — | — | — |
| Azure AI Studio | ✓ | ✓ | ✓ | ✓ |
| Deepinfra | ✓ | ✓ | ✓ | ✓ |
| Fireworks AI | ✓ | ✗ | ✓ | ✓ |
| Groq | ✓ | ✗ | ✓ | ✗ |
| Perplexity | ✗ | ✗ | ✗ | ✗ |
| Replicate | ✓ | — | ✓ | — |
| Together.ai | ✓ | ✓ | ✓ | ✓ |

- **Perplexity** : échoue sur les 8 tests de Table 1 et sur les 4 modèles. Puissance ≈ 1,00 partout (Table 15).
- **Amazon Bedrock** : échoue sur les 4 modèles ; Llama-3 et 3.1 70B les plus éloignés des autres fournisseurs (Fig. 4 gauche).
- Groq et Fireworks : échecs ponctuels sur 3.1 8B (et Groq 3.1 70B).

**Incohérences internes à noter** (le papier ne les commente pas) :
- Table 1 (par distribution de prompts) et Table 16 (combinée Bonferroni) ne coïncident pas toujours. Exemple : Table 1 marque Replicate Llama-3 70B / Wikipedia ✗ (puissance 0,33 dans Table 15 ; < 0,5), alors que Table 16 écrit ✓.
- Table 15 indique Amazon Llama-3 8B / Wikipedia à 0,07, cohérent avec ✓, mais Table 16 écrit ✗ pour Amazon 3 8B après combinaison (échec UltraChat dans Table 1).

**Taille d'effet (§5)** : MMD entre API et référence (10 bootstraps × N = 100m).
- Perplexity Llama-3 8B vs fp32 sur Wikipedia = **0,03**, du même ordre que la distance Llama-3 8B vs GPT-3.5 Turbo (0,03), Phi-3 mini (0,04), OLMo 7B (0,04). Autrement dit, certaines API sont aussi éloignées de la référence qu'un autre modèle.

**Corrélation avec la précision HumanEval (Fig. 4 droite)** : régression 1,824·x + 0,060 ; R² = 0,39 (le texte donne 0,392, la figure 0,394). Corrélation modérée. Les paires où les deux précisions sont < 10 % sont à MMD élevé et écart faible : le MMD capte les "multiples façons d'être faux".

**Coût** : chaque audit < 1 $. Le plus cher (Azure Llama-3 70B) : Wikipedia 0,14 $, HumanEval 0,83 $, UltraChat 0,93 $. Le moins cher (Fireworks Llama-3 8B) : < 0,02 $ pour les trois.

**Divulgations (note 7)** : à la date d'écriture, seuls deux fournisseurs ont annoncé des optimisations qui changent la distribution : **Fireworks AI** (semantic caching) et **Together.ai** (quantization).

**Observations qualitatives (Box 2, 15, Appendix D)** :
- Perplexity produit une distribution d'entropie plus basse que fp32, suggérant un cache ou une mauvaise implémentation de la température (App. D intro).
- nf4 sur Llama 70B : complétions dégénérées ("Course!!!!!!!!") (Box 9).

### 4.7 Autres résultats
- Distance inter-API (Fig. 14–26) : les fournisseurs qui passent le test se regroupent avec fp16 et fp32.
- Llama-3.1 405B (Fig. 26) : écarts plus faibles que sur les modèles antérieurs. DeepInfra déclare servir le 405B en **fp8** (note de Fig. 26).

---

## 5. Audit des APIs commerciales : liste et résultats

| Fournisseur | Modèles audités | Remarques |
|---|---|---|
| Amazon Bedrock | 3 8B, 3.1 8B, 3 70B, 3.1 70B | Échoue sur les 4 modèles |
| Anyscale | 3 8B seulement | Endpoints déprécié en août 2024 pendant la collecte |
| Azure AI Studio | 4 modèles | Passe partout |
| Deepinfra | 4 modèles | Passe partout ; sert le 405B en fp8 |
| Fireworks AI | 4 modèles | Échec sur 3.1 8B ; semantic caching déclaré |
| Groq | 4 modèles | Échec sur 3.1 8B et 3.1 70B |
| Perplexity | 4 modèles | Échoue partout |
| Replicate | 3 8B, 3 70B | Passe selon Table 16 ; Table 1 marque 3 70B / Wikipedia ✗ |
| Together.ai | 4 modèles | Passe partout ; quantization déclarée. Auteur PL co-fondateur de Together (voir Limites) |

Dates de collecte : juillet à août 2024 (Tables 2–6).

---

## 6. Limites

### Limites méthodologiques (papier)
- **Hamming = pseudo-métrique** : MMD = 0 n'implique pas P = Q (App. A.1). Certaines distorsions (fp16) ne sont pas détectées à N = 10m.
- **Puissance dépendante de la paire** : quantification, watermark, finetuning ont des puissances très variables (Table 8, Table 10).
- **Espace caractères** : la puissance baisse en dimension plus grande (Table 10). Il faut N = 50m pour compenser (Table 11).
- **Composite (§5)** : règle pessimiste, la puissance baisse (Table 18).
- **Pas d'information sur la cause** : le test dit qu'il y a une différence, pas laquelle. Les auteurs proposent un examen manuel.
- **Corrélation MMD / précision** modérée (R² ≈ 0,39).
- **Monte Carlo** : puissance estimée sur 100 simulations, p-values sur 1000 (ou 100 pour all-substrings). Estimations bruitées.
- **Calibration** : FPR observé ≈ 0,04–0,06 à α = 0,05 (Table 8). Pas de vérification au-delà.

### Limites du setup
- **Un seul modèle de référence** et des données Llama-3/3.1 de l'été 2024. Les API ont pu changer depuis (le papier ne mesure qu'à une date).
- **Un seul vérificateur** (HumanEval) ; l'effet réel sur la qualité n'est pas mesuré sur les autres tâches.
- **Hypothèse de template** : le papier a fait correspondre les chat templates au nombre de tokens renvoyé par les APIs, mais ne peut pas vérifier les prompts système côté fournisseur.
- **Conflits d'intérêt** : PL (Percy Liang) est co-fondateur de Together AI. Le papier affirme que les résultats n'ont pas été partagés avec Together ou d'autres fournisseurs avant publication.
- **Données publiées** : 1 M de complétions en espace tokens et caractères, pour cinq modèles seulement.

### Limites de la formalisation
- Le test suppose des **échantillons i.i.d.** : les complétions API sont indépendantes les unes des autres (le papier échantillonne un à un). Un cache côté serveur (ex. Fireworks semantic caching) peut violer cette hypothèse (inférence : le papier mentionne le cache sans l'analyser).
- Le papier ne traite pas le cas où les fournisseurs changent le modèle entre deux requêtes d'un même audit.

---

## 7. Code et dataset publiés

- **Dépôt principal** (package Python, code des expériences, dataset) : **https://github.com/i-gao/model-equality-testing**
  - Le papier indique explicitement que ce dépôt contient le package, le code et le dataset de 1 M de complétions (§1, §7, App. B.1).
- Dataset : cinq modèles (Mistral-7B-Instruct-v0.3, Meta-Llama-3-8B-Instruct, Meta-Llama-3.1-8B-Instruct, Meta-Llama-3-70B-Instruct, Meta-Llama-3.1-70B-Instruct), en fp32, fp16, int8, nf4, watermark et chaque API audité.
- Le papier ne donne **pas** d'URL Hugging Face pour le dataset. Les références HF citées sont des sources externes :
  - Cohere, wikipedia-2023-11-embed-multilingual-v3 (source des prompts Wikipedia) : https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3
- Référence externe pour CodeAlpaca : https://github.com/sahil280114/codealpaca

---

## 8. Idées de challenges hackathon et pistes d'implémentation

### 8.1 Pseudo-code du test MMD (noyau de Hamming, permutation)

Pseudo-code fidèle au papier, en Python. Les points marqués (inférence) ne sont pas dans le papier.

```python
import numpy as np

PAD = "\x00"  # token de padding (eq. 7 : y plus court que L est complété)

def pad(y: str, L: int) -> str:
    # tronque ou complète à L (caractères ou tokens)
    return (y + PAD * L)[:L]

def k_hamming(y: str, yp: str, L: int) -> int:
    # eq. (7) : nombre de positions égales
    # (inférence) le papier n'écrit pas de normalisation ; diviser par L
    # ne change pas les p-values (test invariant à l'échelle)
    a, b = pad(y, L), pad(yp, L)
    return sum(ca == cb for ca, cb in zip(a, b))

def mmd2_unbiased(Dp, Dq, L):
    # Dp, Dq : listes de (x, y) ; k(z, z') = 1{x = x'} * k~(y, y')  (section 3)
    def k(z, zp):
        (x, y), (xp, yp) = z, zp
        return k_hamming(y, yp, L) if x == xp else 0.0

    n, m = len(Dp), len(Dq)
    # termes intra : paires distinctes i != j  (inférence : formule eq. 4)
    s_pp = sum(k(Dp[i], Dp[j]) for i in range(n) for j in range(n) if i != j)
    s_qq = sum(k(Dq[i], Dq[j]) for i in range(m) for j in range(m) if i != j)
    s_pq = sum(k(a, b) for a in Dp for b in Dq)
    return s_pp / (n * (n - 1)) + s_qq / (m * (m - 1)) - 2 * s_pq / (n * m)

def permutation_pvalue(Dp, Dq, L, B=1000, seed=0):
    # Algorithme 2 (App. A.2)
    rng = np.random.default_rng(seed)
    obs = mmd2_unbiased(Dp, Dq, L)
    pool = Dp + Dq
    n = len(Dp)
    null = np.empty(B)
    for b in range(B):
        idx = rng.permutation(len(pool))
        A = [pool[i] for i in idx[:n]]
        Bq = [pool[i] for i in idx[n:]]
        null[b] = mmd2_unbiased(A, Bq, L)
    # le papier : proportion des stats permutées >= observée
    # (inférence : beaucoup de code ajoute +1 au numérateur et au dénominateur)
    return float(np.mean(null >= obs))
```

Algorithme 1 (simulation sous H0, si on a accès à P) : même statistique, mais remplacer la permutation par des tirages D_P^(b), D_Q^(b) indépendants depuis P, puis p = proportion des stats nulles ≥ observée.

Test composite (§5) : calculer p1 (contre fp32) et p2 (contre fp16), rejeter si p1 < α **et** p2 < α.

Note de performance (inférence) : la complexité est O(N²·L) par statistique. Pré-calculer la matrice de Gram par prompt rend les permutations beaucoup plus rapides.

### 8.2 Pistes de challenges

1. **"Audit en 250 requêtes"** : fournir un endpoint OpenAI-compatible (vrai ou simulé) et un modèle de référence local. Les équipes doivent détecter une quantification int8/nf4 avec N = 10m, à α = 0,05. Mesure : puissance sur 20 simulations.
2. **"Quel modèle sert-on ?"** : les participants reçoivent des complétions d'un modèle inconnu parmi 5 candidats et doivent identifier le plus proche (MMD min). Reprend Fig. 3 droite.
3. **"Dérive dans le temps"** : un endpoint change de version en cours de route. Détecter le changement avec un test séquentiel (inférence : hors papier, qui traite le cas statique).
4. **"Hamming vs autres noyaux"** : comparer Hamming, Levenshtein, one-hot et L1 sur un nouveau jeu de prompts (code, dialogue). Reproduit Tables 13–14.
5. **"Audit de température"** : détecter un décodage avec τ ≠ 1 (Table 12, puissance ~1,0 pour τ = 0,5 et 1,5).
6. **"Détecteur de caching"** : construire un endpoint qui renvoie la même complétion pour des requêtes répétées, puis le détecter (cf. observation Perplexity sur l'entropie, Box 2).
7. **"Effet sur la qualité"** : sur HumanEval, corréler MMD et précision avec un vérificateur (inférence : étendre à un autre benchmark).

### 8.3 Plan d'implémentation simplifié (inférence)

1. Choisir m = 20 à 25 prompts, N = 10m complétions par modèle (temp 1, L = 50 tokens comme dans le papier).
2. Collecter D_P (référence locale, éventuellement via le dataset publié) et D_Q (API).
3. Calculer MMD^ avec le noyau de Hamming.
4. p-value par permutation (B = 1000).
5. Répéter 10 fois, rejeter l'endpoint si le taux moyen de rejet ≥ 0,5 (règle du papier).
6. Pour un audit de production, utiliser le dataset publié comme référence et compter environ 0,02 à 0,93 $ par audit selon le fournisseur (ordre de grandeur du papier).

---

## 9. Glossaire

- **Test à deux échantillons** : test statistique qui décide si deux échantillons proviennent de la même distribution (ici P et Q).
- **MMD (Maximum Mean Discrepancy)** : distance entre distributions, égale à la différence de leurs embeddings moyens dans un espace de Hilbert défini par un noyau.
- **Noyau (kernel)** : fonction k(z, z') qui mesure la similarité entre deux objets. Son feature map φ envoie les objets dans un espace où le MMD se calcule.
- **Noyau universel** : noyau pour lequel le MMD est nul si et seulement si P = Q. All-substrings et one-hot sont universels ; Hamming ne l'est pas.
- **Pseudo-métrique** : distance qui vérifie la symétrie et l'inégalité triangulaire, mais pour laquelle d(P, Q) = 0 n'implique pas P = Q.
- **Distance de Hamming** : nombre de positions où deux séquences de même longueur diffèrent (ici on compte les positions égales, eq. 7).
- **p-value** : probabilité, sous H0, d'observer une statistique au moins aussi extrême que celle observée.
- **Erreur de type I (FPR)** : rejeter H0 alors qu'elle est vraie. Contrôlée à α.
- **Puissance** : probabilité de rejeter H0 quand H1 est vraie. Ici, moyenne sur des alternatives Q.
- **Permutation (test)** : construction de la distribution nulle en mélangeant les étiquettes des échantillons.
- **Hypothèse composite** : H0 qui regroupe plusieurs distributions de référence (ici fp32 OU fp16).
- **Bonferroni** : correction de comparaisons multiples, qui divise α par le nombre de tests.
- **Quantification** : réduction de la précision des poids (fp32 → fp16 → int8 → nf4) pour réduire coûts et mémoire.
- **Watermark (Kirchenbauer et al.)** : modification des probabilités de tokens pour laisser une signature statistique détectable.
- **Finetuning** : poursuite de l'entraînement d'un modèle sur un nouveau jeu de données.
- **Température** : paramètre de décodage qui aplatit (τ > 1) ou accentue (τ < 1) la distribution des tokens.
- **Sample complexity** : nombre d'échantillons nécessaires pour atteindre une puissance donnée.
- **Goodness-of-fit (test d'adéquation)** : test qui compare des observations à une distribution connue, avec accès à la probabilité P(y|x).
- **Semantic caching** : cache qui renvoie une réponse précédente pour une requête sémantiquement proche (déclaré par Fireworks AI, note 7).
- **Feature map φ** : fonction qui projette un objet dans l'espace de Hilbert associé au noyau.
