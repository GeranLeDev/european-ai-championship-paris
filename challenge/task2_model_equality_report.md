# Task 2 — Model Equality Testing : rapport (protocole, plan de requêtes, simulateur)

_Rédigé le 21/11 ~17h. Aucune requête envoyée au serveur (pas de clé API). Code : `tasks/model_equality/solution/`._

## 1. Protocole API (lu dans le kit)

- Base : `http://34.147.229.55/33-model-equality/v1`, en-tête `X-API-Key: <clé équipe>`.
- `GET /models` → `{"data":[{"id":"ep-01","budget":{"limit":100,"used":k}}, ...]}`. **Gratuit** : sert à lire le budget restant. Liste aussi `ref-01..ref-09`.
- `POST /chat/completions` (format OpenAI) : `{"model":"ep-07","max_tokens":256,"messages":[{"role":"user","content":...}]}`.
  - Réponse : `choices[0].message.content` + `budget` (`used/limit`), sans doute `usage` et `finish_reason` (le collecteur les sauvegarde).
  - `temperature`/`seed` ignorés ; message `system` rejeté ; ≤ 1024 tokens d'entrée, ≤ 256 en sortie.
  - Codes : 200 = OK (1 réponse décomptée) ; 429 + `Retry-After` = trop de requêtes simultanées (attendre) ; 429 sans `Retry-After` = budget épuisé ; 502/503/504 = réessayer. Les requêtes invalides ne sont pas décomptées.
- **Endpoints de référence `ref-01..ref-09`** : `query_reference.py` dit explicitement « no query limit, never scored ». Ils servent un **autre petit modèle**, avec les mêmes déguisements ; les réponses sont dans `practice_answers.csv` (1 famille P ; modèle S = plain, system prompt, persona, watermark, sampling T=1.0 ; modèles distincts T1 = fp8, T2/T3/T4 = LoRA 0.3/0.7/1.5).
  - → **banc d'essai gratuit et étiqueté** : c'est là qu'il faut calibrer les seuils et vérifier les capacités du serveur, **avant** de dépenser le moindre `ep-XX`.
- Budget scoré : 100 réponses par `ep-XX` pour toute la compétition, 4 requêtes simultanées par équipe (clé partagée par toute l'équipe : un seul process de collecte à la fois).
- Soumission : `POST /submit/33-model-equality` (fichier CSV `endpoint,family,model`), 1 toutes les 30 min, score public sur une partie des paires, classement final sur le reste.
- Score : `0,1·max(0,MCC_famille) + 0,9·max(0,MCC_modèle)`, MCC_modèle sur les paires d'une même famille (vraie famille). Réimplémenté dans `met_common.task_score`. `score_practice.py` ne calcule que la partie « même modèle » sur les ref.


### Vérifié sur le serveur (21/11 ~19h, clé d'équipe, **ref uniquement**)
- `GET /models` liste 42 endpoints : `ep-01..ep-33` (limit 100, used 0) et `ref-01..ref-09` (**limit 1 000 000**, compteur séparé). Les requêtes ref ne touchent pas le budget ep (re-vérifié après la collecte, voir §3bis).
- `collect.py probe` sur ref-01 :
  - la réponse ne contient **ni `usage` ni `logprobs`** (ignorés), et `n=2` ne renvoie qu'un seul choix ;
  - le prefill assistant n'est pas continué (l'assistant répond à neuf) ; le multi-tour user/assistant est accepté (1 à 32 messages) ;
  - `system` → HTTP 400 « Invalid request (not counted) » ; `max_tokens=1` fonctionne.
  - → aucun canal latéral : seul le texte est exploitable.
- **Débit mesuré : ~8,5 réponses/min avec 4 requêtes parallèles** (latence ~25-30 s par requête, sorties ≤ 256 tokens).
  - → **la collecte ep complète (3 300 réponses) prendrait ~6,5 h** (2 × ~3,2 h pour 50 + 50). Il faut la lancer tôt.
  - La collecte ref a été réduite à 25 questionnaires × 2 (450 réponses, ~55 min).

## 2. Stratégie de requêtes (100 réponses / endpoint)

### Idée
Les déguisements « même modèle » (system prompt, persona, watermark, décodage) changent surtout **le style et les choix incertains**. Ils ne changent presque pas **la réponse modale à une question courte et contrainte**. Les changements de poids (quantization, LoRA) déplacent ces réponses modales sur les items « limites » (faits obscurs, calcul mental, erreurs caractéristiques).
→ On compare les endpoints sur **les mêmes items courts**, avec une similarité **normalisée par le bruit propre de chaque endpoint** (robuste à la température et au greedy/beam).

### Prompts (fixes, identiques pour tous les endpoints) — `solution/build_prompts.py`
- **50 questionnaires × 16 items = 800 items courts** par endpoint, chacun posé **2 fois** → **100 réponses / endpoint (100 % du budget)**.
- 1 requête = 1 questionnaire numéroté (« Below are 16 numbered questions. Reply with exactly 16 lines… », suivi de « Answers: ») ; ≤ 260 tokens d'entrée, ~80-150 tokens de sortie. On récupère donc **~16 « bits » de signature par requête** au lieu d'un seul.
- Banque d'items (graine fixe) : capitales (125 pays, faciles → obscurs), symboles et numéros atomiques, traductions FR/DE, rimes, associations de mots, nombre de lettres, contraires, trivia (années, auteurs, monnaies), multiplications 2×2 chiffres, additions/soustractions 3 chiffres, « lettre n places après X », « pick a random number/colour/… ».
- Parsing : ligne `N. réponse` → forme canonique (1er nombre ; ou 1er mot de contenu après `=`, `->`, `:`, ` - `). Les mots « de remplissage » d'un endpoint (présents dans > 15 % de ses lignes, ex. « arr » d'un pirate) sont ignorés → robustesse persona.
- Pourquoi 2 répétitions : la 2e réponse sert à estimer **l'auto-accord A(e,e)** de chaque endpoint (son niveau de bruit). Sans ça, impossible de distinguer « autre modèle » de « même modèle, décodage plus bruité ».
- Les prompts ouverts (complétion Wikipédia, Hamming-MMD caractères du papier) ont été testés au simulateur. Ils sont **très sensibles au system prompt** (cos plain/sysprompt 0,63) et peu informatifs à ce budget, donc retirés du plan par défaut (`--f 10 --rf 2` pour les remettre).

### Statistique — `solution/analyze.py`
- A(e,f) = P(réponse canonique de e = celle de f), moyennée sur les items et sur toutes les paires de répétitions ; A(e,e) entre les 2 répétitions de e.
- **cos(e,f) = A(e,f) / √(A(e,e)·A(f,f))**. C'est un noyau « one-hot par item », la version normalisée de la MMD du papier : MMD² = A(e,e)+A(f,f)−2A(e,f). La normalisation enlève l'effet de la température : un endpoint greedy a A(e,e)=1, un endpoint T=1 a un A(e,e) faible.
- Endpoints **déterministes** (auto-accord ≥ 0,95 : greedy/beam) : ils ne passent pas par le seuil. Par définition, ce sont des variantes de décodage d'un modèle de base. On les rattache au groupe dont ils expliquent le mieux les échantillons (A(g,e)/A(e,e)), à égalité au plus grand groupe.
  - Raison : au simulateur, greedy vs beam du même modèle ne s'accordent qu'à ~0,64-0,76 par item. Les erreurs en chaîne dans le questionnaire les font paraître plus éloignés qu'un fp8.
- Clustering hiérarchique (average linkage) sur d = 1−cos :
  - familles : seuil `--t-family` (0,45) ;
  - modèles **dans chaque famille** : seuil `--t-model` (à calibrer sur les ref).
- CSV `endpoint,family,model` cohérent par construction (le label modèle est préfixé par la famille).

## 3. Simulateur local et résultats (MCC avec la formule officielle)

### Montage — `solution/sim.py` (MPS, fp16, ~1,5-4 min par endpoint)
- 3 familles : **A** = Qwen2.5-0.5B-Instruct (12 endpoints), **B** = SmolLM2-360M-Instruct (6), **C** = Qwen2.5-1.5B-Instruct (10, calqué sur ref-01..09 + greedy). Au total **28 endpoints**, chacun avec exactement le budget réel (100 réponses, mêmes prompts).
- Modèles différents :
  - fp8 e4m3 (poids, échelle par canal) ;
  - int4 (groupes de 64, A seulement) ;
  - **LoRA réellement entraîné** (rang 8 ; q, v, o, up, down ; 120 pas sur des réponses auto-générées au style « enjoué ») appliqué à 0,3 / 0,7 / 1,5 (1,0 pour B).
- Même modèle :
  - system prompt (« support client Contoso ») ;
  - persona pirate ;
  - watermark KGW (γ=0,25, δ=2, graine = token précédent) ;
  - T=1,0 ;
  - greedy ;
  - beam 4.
- Décodage par défaut des endpoints « plain » : sampling T=0,7 (hypothèse : le kit dit que les réponses sont échantillonnées, et la ref « T=1.0 » est une variante, donc le défaut ≠ 1,0).
- Vérité terrain cachée derrière des ids `sim-XX` mélangés. Score = `met_common.task_score`, la formule de l'énoncé.

### Résultats
| Jeu | MCC famille | MCC modèle | **Score** |
|---|---|---|---|
| A+B+C (28 endpoints), `--t-model 0.15` (défaut) | 0,935 | 0,326 | **0,387** |
| A+B+C, `--t-model 0.06` | 0,935 | 0,359 | 0,417 |
| A+B (petits modèles, en-tête v1), bootstrap 20×, plateau t=0,16-0,32 | 0,81-0,84 | — | 0,37-0,40 (p10-p90 ≈ 0,22-0,46) |
| C seul (Qwen 1,5B ≈ jeu ref), t=0,06-0,24 | (1 famille) | 0,354 | 0,318 (= 0,9 × 0,354) |
| Référence : familles parfaites, tout « même modèle » | 1 | 0 | 0,100 |
| Oracle « fp8 + LoRA 0,3 fusionnés, le reste parfait » | 1 | — | 0,633 |
| Oracle « fp8 + LoRA 0,3 + LoRA 0,7 fusionnés » | 1 | — | 0,495 |

Similarités cos observées dans C (Qwen 1,5B, auto-accord ~0,69) :
- plain ↔ {fp8, LoRA 0,3, system prompt, T=1,0} : **0,98-1,00** (indiscernables) ;
- plain ↔ {watermark, persona} : 0,96-0,97 ;
- plain ↔ LoRA 0,7 : 0,96, mais LoRA 0,7 ↔ watermark 0,91 → **chevauche les déguisements** ;
- plain ↔ greedy : 0,91 ;
- plain ↔ LoRA 1,5 : **0,74** (détecté nettement) ; int4 (Qwen 0,5B) ↔ plain : ~0,5 (détecté ; il sort même comme famille à part).

Conclusions :
1. Les familles sont faciles (MCC 0,94). L'erreur restante est l'int4 de Qwen 0,5B, si dégradé qu'il sort en famille à part.
2. À 100 réponses, on sépare **LoRA forte et quantization agressive** ; **fp8 et LoRA 0,3 sont indiscernables** de la base, avec tous les noyaux testés : cos, accord brut, MMD one-hot par item, MMD-Hamming caractères (abandonnée).
3. Le pipeline atteint ~0,39-0,42, soit ~65-80 % de l'oracle « fp8/LoRA 0,3 fusionnés » (0,63). Le reste se perd sur LoRA 0,7 et le watermark, qui ont des similarités voisines.
4. Le seuil `--t-model` a des plateaux larges mais **dépend de la taille du modèle** (petits modèles bruités : 0,16-0,32 ; 1,5B : 0,06-0,24). Il faut le **calibrer sur les ref**. Le défaut 0,15 est un compromis.
5. Pondération des items par stabilité (`--gamma`) : gain non robuste au bootstrap, laissée à 0 par défaut.

Reproduire : `$PY analyze.py data/sim_all_answers.jsonl --truth data/sim_all_truth.csv --sweep` ; `$PY eval_boot.py data/simC_answers.jsonl data/simC_answers_truth.csv`. Régénérer : `$PY sim.py --only AB` puis `$PY sim.py --only C --out data/simC_answers.jsonl`, environ 1 h au total. Les deltas LoRA (~5 Go) ont été supprimés ; ils sont réentraînés automatiquement.


## 4. Commandes exactes (quand la clé arrive)

Tout se lance depuis `tasks/model_equality/solution/`, avec `PY=../../../repos/model-equality-testing/.venv/bin/python` (ou n'importe quel Python avec `requests numpy scipy`). La clé est lue dans `MET_API_KEY` et n'est jamais écrite sur disque.

```bash
cd tasks/model_equality/solution
export MET_API_KEY=...            # clé d'équipe
PY=../../../repos/model-equality-testing/.venv/bin/python

# 0. Gratuit : liste des endpoints + budget (GET /models), plan sans rien envoyer
$PY collect.py --endpoints ep --allow-scored --dry-run
# 0b. Gratuit : capacités du serveur sur ref-01 seulement (logprobs ? n ? prefill assistant ? usage ? system rejeté ?)
$PY collect.py probe

# 1. Gratuit et illimité : protocole complet sur ref-01..09, 3 réplicats (3 × 100 réponses par ref)
$PY build_prompts.py --rq 6 --out prompts_ref.jsonl
$PY collect.py --endpoints ref --prompts prompts_ref.jsonl --out data/ref_answers.jsonl --max-per-endpoint 300
for off in 0 2 4; do      # chaque réplicat = exactement le budget réel (2 répétitions)
  $PY analyze.py data/ref_answers.jsonl --prompts prompts_ref.jsonl --prefix ref- --rep-offset $off --max-rep-quiz 2 \
      --truth ../practice_answers.csv --sweep --out data/ref_guess_$off.csv
  $PY ../score_practice.py data/ref_guess_$off.csv
done
$PY eval_boot.py data/ref_answers.jsonl ../practice_answers.csv --prompts prompts_ref.jsonl --prefix ref- --gamma 0
#    -> choisir --t-model sur un PLATEAU (pas le pic), vérifier aussi --gamma 0/1/2.

# 2. SCORÉ (dépense 50 réponses / ep) : 1re répétition de chaque questionnaire
$PY collect.py --endpoints ep --allow-scored --out data/ep_answers.jsonl --max-per-endpoint 50
$PY analyze.py data/ep_answers.jsonl --prefix ep- --max-rep-quiz 1      # contrôle familles (accord brut)

# 3. SCORÉ (50 de plus) : 2e répétition -> budget complet (100/ep)
$PY collect.py --endpoints ep --allow-scored --out data/ep_answers.jsonl --max-per-endpoint 100
$PY analyze.py data/ep_answers.jsonl --prefix ep- --t-model <T_ref> --gamma <G_ref> --out submission.csv --dump data/ep_sims.json

# 4. Soumission (1 / 30 min) ; variantes de seuil ensuite (score public = une partie des paires seulement)
$PY submit.py submission.csv
```

- Le collecteur reprend là où il s'est arrêté (cache `data/*.jsonl`, clé (endpoint, id, repeat)). Il fait au plus 4 requêtes en parallèle, gère 429/Retry-After et 5xx, et s'arrête sur un endpoint épuisé.
- Il plafonne à `--max-per-endpoint` en comptant les réponses déjà en cache. Pour `ep-*`, il refuse de tourner sans `--allow-scored`.
- **Un seul membre de l'équipe lance la collecte `ep`** : la clé et le budget sont partagés, et le cache fait foi.
- `--extra` (champs JSON ajoutés aux requêtes) est interdit avec `--allow-scored` : il sert seulement à tester des options sur ref.

## 5. Risques et points ouverts

1. **fp8 (et LoRA faible) indétectables à ce budget.**
   - Au simulateur, cos(plain, fp8) vaut 0,96-1,00 avec les deux modèles, comme pour les déguisements.
   - Cohérent avec le papier : la puissance contre int8 est de 0,07 avec un N bien plus grand.
   - Conséquence : on accepte de fusionner fp8 dans le groupe « même modèle ». Le coût est d'environ 5 faux positifs par famille ; c'est mieux que de casser le groupe S.
2. **Greedy/beam** : ils diffèrent beaucoup du sampling à cause des erreurs en chaîne dans le questionnaire. On les gère par la règle « déterministe → rattaché au groupe de base ». Les ref n'ont **pas** de greedy/beam (seulement T=1.0) : cette règle n'est validée qu'au simulateur.
3. **Petits modèles = bruit énorme.**
   - Auto-accord : 0,2-0,33 avec Qwen 0,5B et SmolLM 360M, contre 0,70 avec Qwen 1,5B.
   - Si les vrais ep sont petits, le score dépendra surtout de la détection des LoRA fortes et des quantizations agressives.
   - Le score réel ne se calibre que sur les ref, qui sont gratuits : **à faire en premier**.
4. **Transfert ref → ep** : les ref sont « a separate small model ». Le seuil calibré dessus peut ne pas transférer aux familles ep (autre taille, autre bruit).
   - Parade : cos est normalisé par le bruit propre ; on regarde aussi l'histogramme des cos dans chaque famille ep (`--dump`) et l'on vérifie qu'il est bimodal.
5. **Format** :
   - la persona ou un LoRA de style peuvent casser la numérotation. Le parseur gère `N. x`, `Question N … Answer: x`, et les réponses non numérotées s'il y a exactement 16 lignes ; les mots parasites d'un endpoint (« arr ») sont retirés ;
   - un LoRA fort cassait le format au simulateur. C'est un signal en soi (il est détecté), mais cela réduit les items comparables ;
   - avec l'ancien en-tête, Qwen 1,5B omettait la numérotation 2 fois sur 3 → corrigé (en-tête actuel : 16/16 lignes parsées).
6. **Budget irréversible** : 100 % du budget ep est planifié (50 + 50). Il ne reste aucune marge pour une 3e passe ciblée.
   - Alternative possible si les ref montrent que c'est mieux : 40 questionnaires × 2 et 20 réponses gardées pour des prompts ciblés par famille après l'étape 2. Ces prompts doivent rester identiques pour tous les endpoints d'une famille.
7. **Indices gratuits non encore exploités, à vérifier avec `collect.py probe` sur ref** :
   - `usage.prompt_tokens` : s'il inclut le system prompt caché, il trahit directement les endpoints system prompt / persona. `usage.completion_tokens` / nb de caractères donne une empreinte du tokenizer, donc de la famille ;
   - `logprobs` : si le serveur les renvoie, on passe à un test bien plus puissant (comparaison directe des distributions) ;
   - `finish_reason`.
   Le collecteur sauvegarde déjà `usage`, `finish_reason` et `logprobs` éventuels.
8. Le score public ne porte que sur une partie des paires. Il ne faut pas sur-ajuster le seuil sur le leaderboard ; garder la valeur du plateau ref/simulateur.

## 6. Fichiers (`tasks/model_equality/solution/`)
- `build_prompts.py` : banque d'items et `prompts.jsonl` / `prompts_items.json` (50 questionnaires × 2) ; `prompts_ref.jsonl` (× 6 pour les ref).
- `collect.py` : collecteur (cache/reprise, ≤ 4 requêtes parallèles, budget, dry-run, garde-fou ep, `probe`).
- `met_common.py` : parsing et canonicalisation, score officiel (familles + modèles).
- `analyze.py` : similarités, clustering, CSV (`--truth`, `--sweep`, `--gamma`, `--rep-offset`, `--dump`).
- `eval_boot.py` : bootstrap sur les questionnaires pour choisir un seuil sur un plateau.
- `sim.py` : simulateur ; `submit.py` : soumission avec contrôles de format.
- `data/` : réponses simulées et vérités (`sim_all_*`, `simC_*`), logs.
