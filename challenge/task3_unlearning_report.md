# TASK 3 — Unlearning ViT-B/16 : rapport (sam. 21 nov., soir)

Code : `tasks/unlearning/solution/` (local) = `/p/scratch/training2666/geran1/IncomingPHd/unlearning/solution/` (JURECA).
Soumis à ce jour (public) : `sub_r2_g2` 0,2616 (meilleur), `sub_r2_g1` 0,2384, `sub_r2_g0` 0,0798. Résultats les plus récents : §6.4.

## 1. Ce qu'on sait du modèle et du forget set
- `vit_base_patch16_224` timm, LayerScale (init 1e-4), 100 classes, 85,9 M de paramètres. Entrée : `(px/255 − 0,5)/0,5`.
- Les images sont synthétiques, du genre diffusion : figurines, lanternes, phares, bateaux, plantes, moulins…
- `orig` a **mémorisé** le forget set :
  - confiance moyenne 0,9997, CE vs pseudo-label 0,0003 ± 0,0024 ;
  - il résiste aux augmentations légères (crop ≥ 0,6, couleur, flou) : acc 0,99, CE 0,03.
- Ses performances chutent avec des augmentations plus fortes :
  - medium : acc 0,85, CE 0,73 ± 2,07 ;
  - strong : acc 0,52, CE 2,95 ± 4,02.
- **Répartition des classes** (pseudo-labels = argmax d'`orig`) :
  - seulement **45 classes sur 100** sont présentes ;
  - **9 classes ont ~130 images** (0, 6, 7, 8, 29, 30, 31, 33, 38 ; 130 = plafond exact) ;
  - les 36 autres en ont de 46 à 104.
  - Hypothèse plausible : 130 = nombre total d'images d'entraînement par classe, donc **ces 9 classes seraient entièrement oubliées**. Gold ne pourrait alors jamais les prédire.
  - Indice : le biais de la tête d'`orig` est plus élevé sur les classes du forget set.
  - Non vérifiable hors ligne. C'est **la question n°1 à trancher avec le leaderboard** (§5).
- Le serveur évalue aussi des **copies altérées cachées** du forget set (poids 50 %). On les simule par des augmentations légères tenues à part.

## 2. Évaluateur proxy (`scoring.py`)
- Il implémente exactement les formules du docstring de `task_template.py` :
  - `ms` (moyenne/écart-type de la CE vs gold) et `ps` (KL(gold‖sub) à T=2, normalisée par celle d'`orig`), calculés sur 2 groupes (images propres + copies altérées) ;
  - × retain = accord(argmax) × 1/(1+KL(orig‖sub)).
- Labels = pseudo-labels d'`orig` (train acc ≈ 100 %).
- **Gold n'est pas observable.** On utilise 3 hypothèses (`gold_proxy.py`, `scoring.make_hypotheses`) :

| hypothèse | construction | CE gold (moy ± éc.-type) | argmax gold = orig |
|---|---|---|---|
| `medium` | moyenne d'`orig` sur 4 vues augmentées « medium » (crop 0,3–1, couleur, flou, bruit) | 0,65 ± 1,90 | 0,98 |
| `strong` | idem avec des vues « strong » (crop 0,15–0,6) | 2,77 ± 3,95 | 0,79 |
| `cls` | `medium` + suppression totale des 9 classes plafonnées (masse renormalisée, CE + 10) | 3,99 ± 5,13 | 0,65 |

- **Retain proxy** : Imagenette (public, hors distribution), 3000 images en train et 1000 en val.
  - Ce proxy est **pessimiste** : `orig` y est peu confiant, donc l'accord et la KL y bougent facilement.
  - Le vrai control set est in-distribution.
- **Limite importante** : un score proxy élevé sous l'hypothèse qui a servi de cible est en partie circulaire. La matrice croisée méthode × hypothèse est l'information utile.

## 3. Méthodes testées (score final proxy = forget × retain)
Deux niveaux d'entraînement :
- **MLP** (local, CPU) : on n'entraîne que `blocks.11.{norm2, mlp, ls2}` + `norm` + `head`, sur l'état CLS du dernier bloc mis en cache.
  - C'est exact par rapport au modèle complet : seul le token CLS alimente la tête. Différence < 0,005 logit.
  - Script `feat_unlearn.py`. Le Mac était saturé par les autres tâches (swap 25 GB), d'où ce choix.
- **Full** (JURECA A100) : fine-tuning des 12 blocs, `unlearn.py`, environ 3 min pour 600 pas.

Méthodes :
- **TD** (*target distillation*, notre méthode) : KL(q‖sub) à T=2 vers le proxy gold sur le forget set (+ 2 vues augmentées légères), plus α·KL(orig‖sub) sur Imagenette.
- Option `--w-stat` : on fait coller la moyenne et l'écart-type de la CE du batch à ceux de gold. C'est **le terme qui fait gagner le plus** : il crée la queue lourde de pertes que la KL seule ne produit pas.

### 3.1 Baselines (niveau tête ou MLP)
| méthode | medium | strong | cls | remarque |
|---|---|---|---|---|
| orig (baseline) | 0 | 0 | 0 | public = 0,0 |
| adoucissement oracle (logits/τ, seulement sur forget) | ≤ 0,004 | 0,109 (τ=8) | 0,023 | la température seule ne suffit pas |
| Gradient ascent / NegGrad+ (clamp 1 ou 3) | 0 | 0 | 0 | ps = 0 : distribution détruite (CE 5–13) |
| SCRUB (max-KL non borné) | 0 | 0 | 0 | effondrement (CE 58), comme dans notre smoke test |
| TD, tête seule, α=10 | 0,03 | 0,04 | 0,14 | capacité insuffisante |

### 3.2 TD niveau MLP (local), 1000 pas, lr 1e-3, α=10
| cible | w-stat | retain proxy | acc forget | medium | strong | cls |
|---|---|---|---|---|---|---|
| medium | 0 | 0,82 | 0,99 | 0,061 | 0,017 | 0,013 |
| medium | 0,3 | 0,79 | 0,93 | **0,402** | 0,114 | 0,092 |
| strong | 0,1 | 0,69 | 0,62 | 0 | **0,486** | 0,207 |
| cls (α=30) | 0,05 | 0,78 | 0,66 | 0 | 0,215 | **0,617** |
| mix (medium+strong) | 0,1 | 0,75 | 0,85 | 0,164 | 0,313 | 0,189 |
| mixcls (cls+strong) | 0,1 | 0,74 | 0,64 | 0 | 0,463 | 0,486 |

(Le terme par image `--w-ce`, qui force chaque image vers la CE moyenne de ses vues, fait moins bien que `--w-stat`.)

### 3.3 TD full-model (JURECA), 600 pas, lr 2e-5, α=10, bs 64+64
| cible | w-stat | retain proxy (KL) | medium | strong | cls |
|---|---|---|---|---|---|
| medium | 0,3 | 0,94 (0,014) | **0,210** | 0,072 | 0,053 |
| strong | 0,1 | 0,89 (0,054) | 0,020 | **0,302** | 0,185 |
| mix | 0,1 | 0,91 (0,030) | 0,161 | 0,209 | 0,136 |
| cls | 0,05 | 0,89 (0,044) | 0 | 0,268 | **0,276** |

Comparaison des deux niveaux :
- Le full-model **préserve beaucoup mieux le retain** (KL 0,01–0,05 contre 0,06–0,13 au niveau MLP).
- Il colle moins bien à la cible (ps plus bas) à 600 pas. Round 2 (1500 pas, lr 5e-5) : voir §6.

## 4. Candidats produits (validés : chargement strict, float32, finis, 328 MB ; NON soumis)
Dossier `tasks/unlearning/candidates/` :
- `local_mlp_medium.pt` : TD medium + w-stat 0,3. Proxy : medium 0,419 / strong 0,11.
- `local_mlp_strong.pt` : TD strong + w-stat 0,1. Proxy : strong 0,488 / cls 0,21.
  - Vérifié de bout en bout avec `tt.predict` : acc 0,615 sur 200 images du forget set, cohérent avec le cache.
- `local_mlp_mixcls.pt` : TD mixcls + w-stat 0,1. Proxy : strong 0,478 / cls 0,49.
- Les candidats full-model JURECA (meilleur checkpoint par run) sont dans `/p/scratch/.../unlearning/output/runs/<tag>_g<i>/best.pt`.
  - Export : `python solution/make_submission.py output/runs/<run>/best.pt <dst>.pt`.

## 5. Recommandation : utiliser le leaderboard public pour trancher l'hypothèse gold
- Les scores croisés montrent que **chaque cible ne gagne que sous sa propre hypothèse**. Par exemple, la cible strong donne 0 sous medium.
- La meilleure dépense de soumissions est donc de **soumettre 3 candidats contrastés** et de lire le résultat :
  1. cible **medium** (oubli « doux » : gold ≈ un modèle qui généralise bien) ;
  2. cible **strong** (gold nettement moins sûr) ;
  3. cible **mixcls** ou **cls** (9 classes entièrement oubliées).
- Celui qui score le plus indique la bonne hypothèse. Ensuite, on affine la force (w-stat, pas, lr) autour de lui, en full-model sur JURECA.
- Référence : meilleur score public actuel toutes équipes ≈ 0,09.

## 6. Lancer sur JURECA
```bash
ssh juelich
cd /p/scratch/training2666/geran1/IncomingPHd/unlearning     # .venv partagé déjà prêt
# solution/cache/ contient déjà les proxies gold (gold_*.pt), orig_forget_logits.pt et Imagenette (retain_*_u8.npy)
C="--method td --blocks 12 --steps 1500 --lr 5e-5 --bs-f 64 --bs-r 64 --eval-every 100 --n-eval 3500 --n-eval-r 1000"
sbatch solution/jureca_grid.sh <tag> "$C --alpha 10 --target medium --w-stat 0.3" "$C --alpha 10 --target strong --w-stat 0.1" \
       "$C --alpha 10 --target mixcls --select strong --w-stat 0.1" "$C --alpha 3 --target strong --w-stat 0.1"
# 4 configs en parallèle (1 par GPU), logs dans output/<tag>_g<i>.log, checkpoints dans output/runs/<tag>_g<i>/best.pt
python solution/make_submission.py output/runs/<tag>_g1/best.pt output/sub_<tag>.pt   # validation (NE soumet PAS)
```
- Niveau MLP, rapide sur CPU : `python feat_unlearn.py --level mlp --method td --target strong --alpha 10 --lr 1e-3 --w-stat 0.1 --steps 1000 --export ../candidates/x.pt`.
  - Prérequis : `inspect_orig.py`, `gold_proxy.py`, `prep_retain.py`, `cache_feats.py --k 4`, `cache_k1.py`, `cache_mid.py`.

### 6.1 Round 2 full-model (JURECA job 15902109) : 1500 pas, lr 5e-5, bs 64+64
| run | cible | w-stat | α | medium | strong | cls | retain KL / accord |
|---|---|---|---|---|---|---|---|
| r2_g0 | medium | 0,3 | 10 | **0,404** | 0,136 | 0,105 | 0,017 / 0,921 |
| r2_g1 | strong | 0,1 | 10 | 0 | **0,453** | 0,254 | 0,039 / 0,893 |
| r2_g2 | mixcls | 0,1 | 10 | 0 | 0,408 | **0,408** | 0,039 / 0,892 |
| r2_g3 | strong | 0,1 | 3 | 0 | 0,478 | 0,253 | 0,068 / 0,838 |

Le full-model avec plus de pas atteint les scores proxy du niveau MLP, avec une **KL retain 2 à 4 fois plus faible**. Ce sont donc les candidats préférés.

### 6.2 Candidats à sonder sur le leaderboard (classés ; tous validés, NON soumis)
Dossier JURECA : `/p/scratch/training2666/geran1/IncomingPHd/unlearning/output/`
1. `sub_r2_g1.pt` : cible strong. Le meilleur proxy est sous `strong`, et le score reste correct sous `cls` (0,25).
2. `sub_r2_g0.pt` : cible medium (oubli doux). Retain quasi intact (KL 0,017). C'est le seul candidat qui score sous l'hypothèse medium.
3. `sub_r2_g2.pt` : cible mixcls (9 classes supprimées). Couvre à la fois strong (0,41) et cls (0,41).
- Secours en local : `tasks/unlearning/candidates/local_mlp_{strong,medium,mixcls}.pt` (niveau MLP, retain moins bon).

Lecture des 3 scores publics :
- n°2 > n°1 → gold est « sûr ». Explorer w-stat 0,1–0,3 et moins de pas.
- n°1 ≥ n°3 → hypothèse strong. Explorer lr 1e-4, w-stat 0,1–0,3.
- n°3 en tête → les classes sont entièrement oubliées. Pousser cls/mixcls.

### 6.3 Sweeps soumis, en attente dans la file (suivi job → candidat)
- **r3** (job 15902274), 2500 pas, α=10 :
  - g0 : medium, w-stat 0,3, lr 5e-5 ;
  - g1 : strong, w-stat 0,1, lr 1e-4 ;
  - g2 : mixcls (sélection cls), w-stat 0,1 ;
  - g3 : mix (sélection strong), w-stat 0,2.
- **r4** (job 15902279), sweep autour de medium, 1500 pas, lr 5e-5 :
  - g0 : w-stat 0,1 ;
  - g1 : w-stat 1,0 ;
  - g2 : α=3, w-stat 0,3 ;
  - g3 : w-stat 0,3, seed 1.
- **r5** (job 15902280), sweep strong/cls :
  - g0 : strong, w-stat 0,3 ;
  - g1 : strong, α=30 ;
  - g2 : cls, w-stat 0,1 ;
  - g3 : strong, seed 1.
- **Retour leaderboard public** : `sub_r2_g1.pt` (cible strong) = **0,2384** (meilleure autre équipe ≈ 0,09). L'hypothèse strong est donc la plus proche.
  - r4 (sweep autour de medium) a été annulé.
  - Remplacé par deux sweeps autour de r2_g1 (cible strong, bs 64+64, éval toutes les 100 étapes) :
- **r6** (job 15902390) :
  - g0 : w-stat 0,05 ;
  - g1 : w-stat 0,2 ;
  - g2 : lr 3e-5, 2000 pas ;
  - g3 : lr 1e-4, 1000 pas.
  - Toutes les autres configs : α=10, lr 5e-5, 1500 pas, w-stat 0,1.
- **r7** (job 15902391) :
  - g0 : α=5 ;
  - g1 : α=20 ;
  - g2 : p-aug 0,8 (plus de vues altérées) ;
  - g3 : seed 2.
  - Toutes les autres configs : lr 5e-5, 1500 pas, w-stat 0,1.
- Les runs strong de r3 (g1 : lr 1e-4, 2500 pas) et de r5 (g0, g1, g3) sont aussi des candidats autour de strong.
- **Leaderboard public, suite** :
  - `sub_r2_g0` (medium) = **0,0798**, contre 0,2384 pour strong. L'hypothèse strong l'emporte nettement sur medium.
  - Le score de `sub_r2_g2` (mixcls) n'est pas encore connu.
- **Nouvelle file JURECA** : les sweeps sont des tâches dans `IncomingPHd/queue/pending/` (`10_t3_r6`, `11_t3_r7`, `11z_t3_xstrong`, `12_t3_r5`, `13_t3_r3`), et non plus des jobs sbatch.
  - Chaque tâche exporte elle-même `output/sub_<run>.pt` et écrit le classement `collect.py` dans `queue/logs/`.
- **`11z_t3_xstrong`** sonde un oubli encore plus fort.
  - Il calcule un nouveau proxy gold `xstrong` : crop 0,08–0,35, couleur 0,5, flou, bruit ; `augs.xstrong` + `gold_proxy.py --names xstrong`.
  - Puis il entraîne 4 configs vers ce proxy :
    - g0 : w-stat 0,1, α=10, lr 5e-5, 1500 pas ;
    - g1 : w-stat 0,3, α=10, lr 5e-5, 1500 pas ;
    - g2 : w-stat 0,1, α=20, lr 5e-5, 1500 pas ;
    - g3 : w-stat 0,1, α=10, lr 1e-4, 1000 pas.
- **Leaderboard public** : `sub_r2_g2` (mixcls) = **0,2616**, notre meilleur score. Ordre : mixcls > strong 0,2384 >> medium 0,0798. La composante « 9 classes entièrement oubliées » aide.
  - `scoring.make_hypotheses` évalue désormais aussi les hypothèses `mixcls` et `clsstrong` (strong + suppression des 9 classes).
- **Sweeps m1–m3** (tâches `10a`, `10b`, `10c`, terminées 19:30). Base : mixcls, α=10, lr 5e-5, 1500 pas, w-stat 0,1. Scores proxy :

| run | variation | strong | cls | mixcls | clsstrong | KL retain |
|---|---|---|---|---|---|---|
| m1_g0 | w-stat 0,05 | 0,432 | 0,389 | 0,454 | 0,288 | 0,033 |
| m1_g1 | w-stat 0,2 | 0,383 | 0,390 | 0,448 | 0,317 | 0,047 |
| **m1_g2** | **α=5** | 0,419 | 0,436 | **0,490** | 0,327 | 0,055 |
| m1_g3 | α=20 | 0,377 | 0,360 | 0,424 | 0,281 | 0,027 |
| m2_g0 | lr 1e-4, 1000 pas | 0,396 | 0,392 | 0,453 | 0,299 | 0,049 |
| m2_g1 | lr 3e-5, 2000 pas | 0,395 | 0,400 | 0,464 | 0,309 | 0,037 |
| m2_g2 / g3 | seeds 1 / 2 (= config r2_g2) | 0,408 / 0,404 | 0,408 / 0,413 | 0,474 / 0,476 | 0,316 | 0,040 / 0,038 |
| m3_g0 | cible clsstrong | 0,086 | 0,222 | 0,184 | 0,447 | 0,069 |
| m3_g1 | clsstrong, α=20, w-stat 0,05 | 0,213 | 0,318 | 0,333 | 0,419 | 0,042 |
| **m3_g2** | **cible cls** | 0,240 | **0,502** | 0,417 | 0,322 | 0,041 |
| **m3_g3** | **cls, α=20, w-stat 0,05** | 0,273 | 0,463 | 0,418 | 0,284 | **0,024** |

- **Top 3 transmis pour sondage** (`output/sub_*.pt`) :
  1. `sub_m1_g2.pt` (α=5) ;
  2. `sub_m3_g2.pt` (cible cls) ;
  3. `sub_m3_g3.pt` (cls, retain prudent).
### 6.4 Suite (soir du 21/11)

**Scores publics.** Ils sont monotones en α (poids de préservation du retain) : plus α est bas, mieux c'est.

| soumission | cible | α | pas | public |
|---|---|---|---|---|
| r2_g0 | medium | 10 | 1500 | 0,0798 |
| m3_g2 | cls | 10 | 1500 | 0,2046 |
| m3_g3 | cls | 20 | 1500 | 0,2080 |
| r2_g1 | strong | 10 | 1500 | 0,2384 |
| r2_g2 | mixcls | 10 | 1500 | 0,2616 |
| m1_g2 | mixcls | 5 | 1500 | 0,2736 |
| ma_g1 | mixcls | 3 | 1500 | 0,2806 |
| **ma_g0** | mixcls | 2 | 1500 | **0,2893** (meilleur) |

**fp16** (`fp16check.py`) : aucun impact.
- Pas de NaN ni d'Inf.
- Écart des logits avec l'autocast fp16 : max |Δlogit| ≤ 0,024. Aucune classe prédite ne change.
- Scores proxy identiques.

**Soupes** (`soup.py`) : pas de gain.
- Soupe greedy sur 12 runs mixcls : 0,507, contre 0,506 pour le meilleur run seul.
- Soupe uniforme : 0,473, un oubli plus doux.
- Soupe de seeds à α=2 : 0,515, contre 0,537 pour un seul run à 2500 pas.

**α bas (tâche 06)** :
- α=1 : mixcls 0,491, KL 0,112, accord 0,803. Le retain se dégrade ; à éviter.
- α entre 1,5 et 2 : environ 0,50–0,51.

**Le levier principal est le nombre de pas** (tâche 08). Le retain s'améliore en même temps : KL en baisse, accord en hausse.

| run | α | pas | mixcls | strong | cls | KL retain | accord |
|---|---|---|---|---|---|---|---|
| al2_g0 | 2 | 2500 | 0,537 | 0,458 | 0,486 | 0,069 | 0,830 |
| st_g0 | 2 | 4000 | 0,574 | 0,479 | 0,524 | 0,063 | 0,840 |
| st_g1 | 1,5 | 4000 | **0,577** | 0,480 | 0,526 | 0,072 | 0,840 |
| **st_g3** | 3 | 4000 | 0,574 | 0,480 | 0,524 | **0,052** | **0,851** |

- **Candidats recommandés** : `output/sub_st_g3.pt` (meilleur compromis), puis `sub_st_g1.pt`.
- Tâche en file : `09_t3_long` (runs lg_g0..g3).
  - g0 : α=3, 8000 pas ;
  - g1 : α=2, 8000 pas ;
  - g2 : α=3, lr 1e-4, 4000 pas ;
  - g3 : α=3, 6000 pas, seed 1.
**Nouveaux scores publics** : `sub_st_g3` = **0,3043** et `sub_st_g1` = 0,3025. Plus de pas vaut mieux qu'un α plus bas.

**Dernier sweep, `09_t3_long`** (cible mixcls, w-stat 0,1) :

| run | α | pas (meilleur à) | lr | mixcls | strong | cls | KL retain | accord |
|---|---|---|---|---|---|---|---|---|
| **lg_g0** | 3 | 8000 (7000) | 5e-5 | **0,618** | 0,513 | 0,563 | **0,047** | **0,861** |
| lg_g1 | 2 | 8000 (6800) | 5e-5 | 0,618 | 0,519 | 0,556 | 0,057 | 0,854 |
| lg_g3 | 3 (seed 1) | 6000 (5200) | 5e-5 | 0,605 | 0,501 | 0,551 | 0,050 | 0,861 |
| lg_g2 | 3 | 4000 | 1e-4 | 0,583 | 0,485 | 0,532 | 0,059 | 0,835 |

**Recommandation finale** : `output/sub_lg_g0.pt`, puis `sub_lg_g1.pt`.
- Config : `--method td --blocks 12 --target mixcls --alpha 3 --lr 5e-5 --w-stat 0.1 --steps 8000 --bs-f 64 --bs-r 64`.
- Le gain plafonne vers 7000 pas.
- Le nœud est ensuite rendu à la Task 1 : plus aucune tâche t3 en file.

- Pour récapituler et exporter :
  ```bash
  cd /p/scratch/training2666/geran1/IncomingPHd/unlearning && source .venv/bin/activate
  python solution/collect.py
  cd solution && python make_submission.py ../output/runs/<run>/best.pt ../output/sub_<run>.pt
  ```
  `collect.py` donne, par run, la config et le meilleur score proxy sous chaque hypothèse.

### 6.4 Sweeps mixcls / cls (sam. 21 nov., 19:17–19:50) et calage du proxy sur le public
Score public de `sub_r2_g2` (mixcls) : **0,2616**. Vérité terrain disponible : 3 points (r2_g0 0,0798 ; r2_g1 0,2384 ; r2_g2 0,2616).

**Runs terminés** (full-model, bs 64+64 ; mesuré sur le proxy ; colonnes = hypothèse gold) :

| run | cible | w-stat | α | lr / pas | strong | cls | mixcls | clsstrong | retain KL / accord |
|---|---|---|---|---|---|---|---|---|---|
| m1_g0 | mixcls | 0,05 | 10 | 5e-5 / 1500 | 0,432 | 0,389 | 0,454 | 0,288 | 0,033 / 0,905 |
| m1_g1 | mixcls | 0,2 | 10 | 5e-5 / 1500 | 0,383 | 0,390 | 0,448 | 0,317 | 0,047 / 0,870 |
| m1_g2 | mixcls | 0,1 | 5 | 5e-5 / 1500 | 0,419 | 0,436 | **0,490** | 0,327 | 0,055 / 0,858 |
| m1_g3 | mixcls | 0,1 | 20 | 5e-5 / 1500 | 0,377 | 0,360 | 0,424 | 0,281 | 0,027 / 0,905 |
| m2_g0 | mixcls | 0,1 | 10 | 1e-4 / 1000 | 0,396 | 0,392 | 0,453 | 0,299 | 0,049 / 0,860 |
| m2_g1 | mixcls | 0,1 | 10 | 3e-5 / 2000 | 0,395 | 0,400 | 0,464 | 0,309 | 0,037 / 0,890 |
| m2_g2 | mixcls (seed 1) | 0,1 | 10 | 5e-5 / 1500 | 0,408 | 0,408 | 0,474 | 0,316 | 0,040 / 0,889 |
| m2_g3 | mixcls (seed 2) | 0,1 | 10 | 5e-5 / 1500 | 0,404 | 0,413 | 0,476 | 0,316 | 0,038 / 0,887 |
| m3_g0 | clsstrong | 0,1 | 10 | 5e-5 / 1500 | 0,086 | 0,222 | 0,184 | **0,447** | 0,069 / 0,840 |
| m3_g1 | clsstrong | 0,05 | 20 | 5e-5 / 1500 | 0,213 | 0,318 | 0,333 | 0,419 | 0,042 / 0,878 |
| m3_g2 | cls | 0,1 | 10 | 5e-5 / 1500 | 0,240 | **0,502** | 0,417 | 0,322 | 0,041 / 0,873 |
| m3_g3 | cls | 0,05 | 20 | 5e-5 / 1500 | 0,273 | 0,463 | 0,418 | 0,284 | 0,024 / 0,918 |
| r6_g0 | strong | 0,05 | 10 | 5e-5 / 1500 | 0,410 | 0,232 | 0,335 | 0,221 | 0,034 / 0,900 |
| r6_g1 | strong | 0,2 | 10 | 5e-5 / 1500 | **0,460** | 0,250 | 0,370 | 0,268 | 0,043 / 0,883 |
| r6_g2 | strong | 0,1 | 10 | 3e-5 / 2000 | 0,450 | 0,254 | 0,367 | 0,252 | 0,036 / 0,899 |
| r6_g3 | strong | 0,1 | 10 | 1e-4 / 1000 | 0,436 | 0,246 | 0,356 | 0,244 | 0,049 / 0,870 |

(La colonne medium vaut 0 partout. Les seeds 1 et 2 de la config de `r2_g2` redonnent les mêmes scores proxy à ±0,005 : le bruit de seed est faible.)

**Quelle colonne proxy prédit le public ?** Outils : `dump_logits.py` (logits d'éval de chaque checkpoint, sur JURECA) puis `fit_public.py` (hors ligne, CPU) : pour chaque hypothèse gold, on compare le proxy des 3 runs soumis aux 3 scores publics, à un facteur d'échelle `s` près (public ≈ s × proxy).
- Aucune colonne d'origine ne colle seule : `strong` classe r2_g1 devant r2_g2 (faux), `cls` et `medium` sont loin (résidu relatif 0,19 et 1,06 ; `strong` 0,10).
- Les meilleures hypothèses combinent **base strong + suppression partielle ou totale des 9 classes** (résidu 0,004 à 0,05) : `split:medium:6:mix:medium+strong`, `mix:cls:strong:{6,10,15,20}+medium`, `mix:cls:strong:{3..20}+strong`, `split:medium:6:strong`.
  - Syntaxe (`scoring.resolve`) : `cls:<base>:<off>` = base + suppression des 9 classes avec un décalage de CE de `off` nats ; `mix:A+B` = moyenne ; `split:<base>:<off>:<autre>` = suppression seulement pour les images des 9 classes, `<autre>` pour le reste.
- **Limite** : 3 points pour 2 à 3 paramètres. Le décalage de CE (6 ou 20 nats) n'est pas identifiable : il change seulement l'échelle `s` (0,6 à 1,2). L'ajustement dit quelles hypothèses sont exclues (medium, cls sur base medium seule), pas laquelle est vraie.
- Sous les 10 meilleures hypothèses, prédiction de score public (s × proxy) :
  - runs mixcls : plage étroite, 0,25–0,28. `m1_g2` est le plus haut sous chacune (0,266–0,281), contre 0,258–0,270 pour `r2_g2` (mesuré 0,2616). Gain attendu faible, de l'ordre de +0,01.
  - runs clsstrong (`m3_g0`, `m3_g1`) : prédictions très dispersées (0,02–0,27) : ce sont des sondes informatives mais risquées.
  - runs cls purs (`m3_g2`, `m3_g3`) : 0,18–0,25, sous le meilleur actuel.
- Conclusion : les variations d'hyperparamètres autour de mixcls ne suffiront pas. Le gain possible vient d'une cible mieux placée dans la famille « strong + 9 classes supprimées », d'où les sweeps ci-dessous.

**Sweeps lancés à 19:50** (file `queue/pending`, α=10, lr 5e-5, 1500 pas, w-stat 0,1 ; exports `output/sub_<run>.pt`) :
- `rx` (`11b_t3_xstrong`) : cible xstrong (CE gold 5,62 ± 4,94), 4 configs, voir §6.3.
- `x1` : `cls:strong:6`, `cls:strong:15`, `mix:cls:strong:10+strong`, `mix:cls:medium:10+cls:medium:10+strong`.
- `x2` : `cls:xstrong:10`, `mix:cls:xstrong:10+strong`, `mix:cls:strong:10+xstrong`, `mix:cls:medium:10+xstrong`.
- `x3` : `split:medium:6:mix:medium+strong`, `mix:cls:strong:6+medium`, `split:medium:6:strong`, `mix:cls:strong:15+medium`.
- `x4` : `mix:cls:strong:20+strong`, `mix:cls:strong:15+strong`, `split:strong:3:medium`, `cls:strong:20`.
- Puis `r7`, `r5`, `r3` (strong, priorité basse).
- Pour classer : `rsync` de `output/logits/` vers `solution/cache/logits/`, puis `python fit_public.py`. Ajouter chaque nouveau score public dans `PUBLIC` (en tête de `fit_public.py`) : chaque point supplémentaire réduit la famille d'hypothèses.

**Sweep `rx` (cible xstrong), terminé à 19:54** (mesuré, proxy) :

| run | w-stat | α | lr / pas | strong | cls | mixcls | clsstrong | xstrong | retain KL / accord |
|---|---|---|---|---|---|---|---|---|---|
| rx_g0 | 0,1 | 10 | 5e-5 / 1500 | 0,167 | 0,093 | 0,181 | 0,334 | 0,506 | 0,081 / 0,838 |
| rx_g1 | 0,3 | 10 | 5e-5 / 1500 | 0,107 | 0,035 | 0,110 | 0,325 | 0,510 | 0,090 / 0,831 |
| rx_g2 | 0,1 | 20 | 5e-5 / 1500 | 0,182 | 0,109 | 0,197 | 0,324 | 0,483 | 0,059 / 0,860 |
| rx_g3 | 0,1 | 10 | 1e-4 / 1000 | 0,159 | 0,089 | 0,173 | 0,326 | 0,494 | 0,090 / 0,813 |

- L'oubli xstrong seul (sans suppression de classes) score bas sous strong, cls et mixcls, et dégrade le retain proxy : pas un candidat. Logits non encore dumpés, donc pas de prédiction `fit_public.py`.

**État de la file à 19:59** : les workers `ipwork1` et `ipwork2` ont été annulés à 19:54:51 (pas par l'agent Task 3) ; `x1` a été interrompu au démarrage et remis en attente. Il reste un worker (`ipwork3`), occupé par `01_t1_rect_train`. File restante : `02_t1_rect_test`, `11c_t3_x3`, `11d_t3_x4`, `11g_t3_x1`, `12a_t3_dump2`, `30_t1_rect_fine_train`. `x2`, `r7`, `r5`, `r3` sont rangés dans `queue/parked_t3/`.

**4e point public (20:15)** : `sub_m1_g2` = **0,2736** (nouveau meilleur ; valeur lue dans le commentaire de `queue/pending/11a_t3_alpha.sh`). `fit_public.py` l'avait prédit entre 0,266 et 0,281.
- Réajustement sur 4 points : la meilleure hypothèse reste `split:medium:6:mix:medium+strong` (résidu relatif 0,003, échelle 0,58 ; prédit 0,0795 / 0,2395 / 0,261 / 0,274 pour 0,0798 / 0,2384 / 0,2616 / 0,2736). Lecture : les 9 classes plafonnées sont supprimées, les autres images se comportent comme un mélange medium + strong.
- Viennent ensuite `mix:cls:xstrong:<off>+medium` (0,015), `mix:cls:strong:6+medium` (0,020), `split:medium:6:strong` (0,025). `medium`, `mix:medium+strong` et les `cls` sur base medium seule restent exclus (résidu > 0,15).
- L'échelle 0,58 de la meilleure hypothèse plafonne le gain : même un proxy de 0,55–0,60 sous cette hypothèse ne donnerait que 0,32–0,35 en public (supposé).
- File à 20:20 : un seul worker (règle des organisateurs : 1 job Slurm pour toute l'équipe). Ordre : tâche T1 en cours, puis `11a_t3_alpha` et `11b_t3_alpha2` (sweeps mixcls à α bas, ajoutés par le coordinateur), `11c_t3_x3`, `11d_t3_x4`. `x1` est rangé dans `queue/parked_t3/`.

**Sweep `x3` (cibles = hypothèses les mieux ajustées), terminé à 21:52** (mesuré, proxy, pas 1500 ; α=10, lr 5e-5, w-stat 0,1) :

| run | cible | strong | cls | mixcls | sous sa propre cible | retain KL / accord |
|---|---|---|---|---|---|---|
| x3_g0 | `split:medium:6:mix:medium+strong` | 0,415 | 0,390 | 0,431 | **0,511** | 0,042 / 0,874 |
| x3_g1 | `mix:cls:strong:6+medium` | 0,430 | 0,326 | 0,403 | 0,427 | 0,033 / 0,897 |
| x3_g2 | `split:medium:6:strong` | 0,443 | 0,358 | 0,435 | 0,493 | 0,053 / 0,861 |
| x3_g3 | `mix:cls:strong:15+medium` | 0,217 | 0,296 | 0,305 | 0,309 | 0,041 / 0,871 |

- La nouvelle syntaxe de cible (`cls:` / `mix:` / `split:`) fonctionne en entraînement complet.
- **Incident** : « Disk quota exceeded » à 21:51 sur `/p/scratch` pendant la fin de `x3`. Conséquences : pas de `hist.json` pour x3_g0–g2 (les `best.pt` sont complets, les scores ci-dessus viennent des logs), un seul export (`sub_x3_g2.pt`), dump des logits échoué. L'écriture refonctionnait à 21:53 ; cause non identifiée (le quota du projet affiché par `jutil` est loin d'être atteint).
- Prédiction publique de `x3_g0` (logits recalculés en local, cohérents avec le log JURECA) : **0,296** sous la meilleure hypothèse, 0,28–0,30 sous les hypothèses `split`, mais 0,23–0,25 sous les hypothèses `mix:cls:xstrong+medium`. Plage 0,23–0,30 : c'est une sonde qui départage ces deux familles, pas un record assuré à α=10.
- `x4` et `x1` n'ont pas tourné (rangés dans `queue/parked_t3/` par le coordinateur).
- Parallèlement, les sweeps du coordinateur montrent que le public monte quand α baisse (commentaires des scripts de file : α=5 0,2736 ; α=3 0,2806 ; α=2 0,2893) et que 2500 pas aident sur le proxy.
- Suite lancée : `13_t3_x5` = meilleures cibles ajustées avec α bas (`split:medium:6:mix:medium+strong` à α=2 et α=1, `split:medium:6:strong` à α=2, `mix:cls:strong:6+medium` à α=2), précédé de `12_t3_dump3` (dump des logits de ma/mb/al/x3/rx).

**Candidats rapatriés et validés en local** (chargement strict, float32, valeurs finies, 328 Mo ; NON soumis), dans `tasks/unlearning/candidates/` :
1. `sub_m1_g2.pt` : mixcls, α=5. Le plus haut sous toutes les hypothèses ajustées. Risque : retain proxy plus bas (KL 0,055, accord 0,858) que `r2_g2` (0,039 / 0,892).
2. `sub_m1_g0.pt` : mixcls, w-stat 0,05. Prédit 0,254–0,274, donc équivalent à `r2_g2` ; retain proxy meilleur (KL 0,033, accord 0,905).
3. `sub_m3_g3.pt` : cls pur, α=20, meilleur retain (KL 0,024, accord 0,918). Prédit 0,19–0,25 : sonde pour mesurer l'effet de la suppression seule, pas un candidat au record.
4. `sub_x3_g0.pt` (ajouté à 22:00, copie de `runs/x3_g0/best.pt`, md5 identique) : cible `split:medium:6:mix:medium+strong`, α=10. Prédit 0,23–0,30 selon l'hypothèse.

## 7. Risques et suite
- **L'hypothèse gold est inconnue.** C'est le risque principal, d'où la stratégie du §5.
  - Un `ms` élevé exige de reproduire la queue de pertes (quelques images très fausses). Si gold est plus « sûr » que notre proxy medium, il faut réduire w-stat et le nombre de pas.
- **Gate 70 %** : l'accuracy held-out n'est pas mesurable sans données in-distribution.
  - L'accord avec `orig` sur Imagenette reste à 0,89–0,94 (full-model) et 0,77–0,85 (MLP), sur des données OOD où `orig` hésite déjà.
  - Le risque est réel pour l'hypothèse cls : si le contrôle contient les 9 classes, l'accuracy perd ~9 %. On reste quand même > 70 % si `orig` est vers 90 %.
- **Collatéral par classe** : les cibles strong et cls font baisser l'acc forget à 0,6. Cela peut déborder sur des images non vues des mêmes classes (contrôle), ce que le proxy Imagenette ne voit pas.
  - Prochaine étape : mesurer l'accord sur des vues *strong* du forget set tenues à part, par classe.
- **Retain proxy plus proche de la distribution** :
  - générer des images in-distribution par inversion du modèle (DeepInversion sur `orig`), surtout pour les 55 classes absentes du forget set ;
  - ou utiliser un dataset de scènes/objets synthétiques.
- **Copies altérées** : la nature exacte des altérations serveur est inconnue. On entraîne avec 50 % de vues légères ; ajouter JPEG et redimensionnement.
- Pistes non testées faute de temps :
  - SCRUB+R borné (max-step clampé + rewind sur la statistique de CE) ;
  - ensembles / moyennes de poids de plusieurs runs TD (gold est une moyenne de modèles : un *weight averaging* adoucirait les sorties de façon naturelle) ;
  - LoRA.

## 8. Reprise nuit du 21 au 22 nov. (00:20 →) : ajustement à 12 points, extrapolation de poids, nouvelles sondes
Objectif fixé : 0,40 public. Point de départ : `lg_g0` 0,3045.

### 8.1 Ajustement sur les 12 scores publics (`fit12.py`, `predict12.py`) — mesuré
- Famille continue de 19 440 hypothèses : gold = mélange pondéré des proxies `medium`/`strong`/`xstrong` (pas de 0,25), décalage de logit `dcap` sur le pseudo-label des images des 9 classes plafonnées, `doth` sur les autres, statistiques de CE poolées ou par image, retain = accord^a / (1 + c·KL), échelle `s` libre (public ≈ s × proxy).
- Qualité : RMS 0,004 sur les 12 points ; validation leave-one-out RMSE 0,007 (écart-type des scores publics : 0,066). L'ajustement prédit donc bien à l'intérieur de la famille TD.
- Meilleures hypothèses : 0,5–0,75 `medium` + 0,25–0,5 `xstrong` (gold bimodal : majorité d'images faciles + queue très dure), CE gold ≈ 2,7–3,5 ± 4,5 ; **suppression des 9 classes faible** (`dcap` 1–3 nats ; `dcap` 12–20 = suppression totale : RMS 0,012–0,017). Même conclusion si on supprime les 9 colonnes pour toutes les images (`fit_rm9.py` : `dcap` 2 → RMS 0,005 ; 12–20 → 0,020).
- Échelle `s` = 0,46–0,61 pour toutes les bonnes hypothèses. `lg_g0` sous la meilleure : ms 0,91, ps 0,76, retain proxy 0,82 → proxy 0,60 × 0,51 = 0,304. **Plafond estimé de la méthode actuelle : ~0,31–0,33** (supposé : `s` représente l'écart par image entre nos proxies et le vrai gold, que l'entraînement ne peut pas combler).
- Limite : 8 des 12 points sont des runs mixcls. La répartition de la perte entre images des 9 classes et les autres est mal identifiée. `lg_g0` met presque toute la perte sur les 9 classes (CE 8,6, acc 0,06) et très peu ailleurs (CE 0,35, acc 0,90) ; les hypothèses ajustées disent l'inverse (CE ≈ 3,2–3,8 sur les 9 classes, 2,5–3,1 ailleurs).

### 8.2 Arithmétique de poids sans entraînement (`extrap.py`, GPU du login) — mesuré (logits réels), prédiction supposée
θ = orig + λ·(lg_g0 − orig). Prédiction publique moyenne [min, max] sur les 30 meilleures hypothèses :

| λ | 0,5 | 0,75 | 1,0 | 1,15 | 1,3 | 1,5 |
|---|---|---|---|---|---|---|
| prédit | 0,063 | 0,223 | 0,310 (mesuré 0,3045) | 0,305 [0,275–0,339] | 0,291 [0,251–0,355] | 0,261 |
| accord retain proxy | 0,924 | 0,891 | 0,861 | 0,844 | 0,832 | 0,803 |

Pas de gain attendu ; λ > 1 dégrade le retain proxy. λ = 1,3 n'a d'intérêt que comme sonde d'information.

### 8.3 Proxies gold « jamais vu » (leave-one-out) — mesuré
- Sonde linéaire 5 plis sur les features d'`orig` à l'entrée du bloc 8 (`loo_proxy.py`) : acc 0,45, corrélation nulle avec les proxies d'augmentation. Couche trop précoce, inutilisable.
- Sonde 5 plis sur des features **DINOv2 ViT-B/14** (modèle public indépendant ; `ext_feats.py`, `loo_ext.py`) : acc 0,89 (9 classes plafonnées 0,78 ; autres 0,94). Les classes plafonnées forment des groupes confusables : {6, 7, 8} et {29, 30, 31, 33} (acc 0,62–0,88, erreurs à l'intérieur du groupe) ; la classe 0 est à 1,00.
- Ajoutée comme base à l'ajustement (`FIT_BASES=medium,strong,xstrong,dinov2b,dino_t3`) : RMS et échelle inchangés (0,004 / 0,5). Les 12 points ne discriminent pas la structure par image.

### 8.4 Sondes lancées (tâche `06_t3_fit`, départ à chaud depuis `lg_g0`, 3000 pas, α=3, lr 5e-5, w-stat 0,1)
`unlearn.py` : nouvelles options `--init` (départ à chaud), `--retain {imagenette,cutmix,both}` (retain in-distribution = CutMix de deux images forget, cible `orig`), `--dump` (logits du meilleur checkpoint, écriture atomique). `make_gold.py` matérialise une hypothèse ajustée en cible.

| run | cible / variante | CE gold visée (9 classes / autres) |
|---|---|---|
| fa_g0 | `fitA` = 0,75 medium + 0,25 xstrong, dcap 3, doth 2 | 2,75 ± 4,49 (3,17 / 2,53) |
| fa_g1 | `fitB` = 0,5 medium + 0,5 xstrong, dcap 1 | 3,32 ± 4,64 (3,76 / 3,09) |
| fa_g2 | `fitC` = 0,5 medium + 0,25 strong + 0,25 xstrong, dcap 2 sur les 9 colonnes, doth 1 | 3,00 ± 4,49 (3,46 / 2,77) |
| fa_g3 | mixcls + retain moitié Imagenette, moitié CutMix d'images forget | — |

- **Incident 00:45** : la connexion SSH (ControlMaster) vers JURECA a expiré ; relance manuelle + MFA nécessaire. La tâche était déjà en file.

### 8.5 Sonde décisive sans entraînement : décalage du biais de la tête sur les 9 classes plafonnées (`bias_probe.py`, `bias_shift.py`)
- Constat (mesuré) : parmi les 12 runs soumis, le public monte de façon monotone avec la suppression des 9 classes (acc sur leurs images 0,33 → 0,06 quand le public passe de 0,26 à 0,30). Mais tous les runs s'arrêtent à CE moyenne ≈ 3 ± 5, valeur imposée par notre propre terme `w-stat`. La zone « suppression plus forte » (CE des 9 classes ≫ 9) n'a jamais été explorée : l'ajustement du §8.1 ne peut pas la juger.
- Sonde : `head.bias[c] −= δ` pour les 9 classes. Effet exact sur les logits (colonne décalée), donc prédictible sans calcul GPU. Prédiction publique de `lg_g0 + δ` (supposée) :

| famille d'hypothèses (RMS d'ajustement) | δ=0 | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| suppression faible, top-30 du §8.1 (0,004–0,006) | 0,315 | 0,286 | 0,221 | 0,151 | 0,088 | 0,009 |
| suppression totale, décalage propre 12–20 nats (0,017) | 0,311 | 0,336 | 0,364 | 0,393 | 0,42 | 0,38–0,47 |
| suppression totale des 9 colonnes (0,020) | 0,304 | 0,344 | 0,387 | 0,436 | 0,48 | 0,48–0,59 |

- Les prédictions « suppression totale » n'incluent pas le coût retain in-distribution : si le contrôle contient ces 9 classes (~9 % des images), leur accuracy et leur KL se dégradent avec δ (ordre de grandeur : facteur 0,75–0,9 à δ=8). Le proxy Imagenette ne le voit pas.
- Candidats (validés : chargement strict, float32, finis ; seul `head.bias` diffère ; logits réels conformes à 0,18 près) : `candidates/sub_lg0_bias4.pt` (référence 0,3045) et `candidates/sub_m1g2_bias4.pt` (référence 0,2736). Lecture : au-dessus de la référence → suppression totale, poursuivre avec δ = 8 ; nettement en dessous → hypothèse close.
- Calibration affine exacte de `lg_g0` (température τ, biais des 9 classes, biais des 36 classes partielles) sous les 30 meilleures hypothèses : `lg_g0` est à un optimum local (meilleur : τ = 1,1, +0,006). Aucun gain gratuit dans le monde « suppression faible ».
- Outils locaux ajoutés : `cache_mid_ckpt.py` (états CLS du dernier bloc pour n'importe quel checkpoint) et `tail_ft.py` (ré-entraînement CPU de la queue d'un run existant vers une nouvelle cible, exact vis-à-vis du modèle complet). Entraînement complet sur MPS mesuré : ~3 s/pas (bs 32+32), donc inutilisable.

### 8.6 Résultat de la sonde et ajustement à 13 points (01:45–02:10)
- **Mesuré** : `sub_lg0_bias4` = **0,2348** public (référence `lg_g0` 0,3045). La suppression totale des 9 classes chez gold est écartée (`dcap` 12–20 : RMS 0,016–0,025).
- Le point était prédit 0,20–0,22 (LOO : 0,202–0,216). L'erreur du fit hors de la zone explorée est donc d'environ ±0,03.
- Ajustement à 13 points : RMS 0,005, LOO RMSE 0,008–0,010. Une seule famille reste en tête : gold ≈ **0,5 medium + 0,5 xstrong, dcap ≈ 1, doth ≈ 0** (= cible `fitB`), s = 0,53–0,62, CE gold ≈ 3,3 ± 4,6. Gold ne traite presque pas les 9 classes à part ; il est bimodal sur toutes les images.
- Transformations affines des logits de `lg_g0` (température, biais des 9 classes relevé de +1 à +4, biais des 36 classes partielles) : toutes prédites neutres ou négatives (biais +1 : 0,307 contre 0,313). Pas de candidat sans entraînement dans cette direction.
- **Ré-entraînement local de la queue de `lg_g0`** (`tail_ft.py`, α=30, lr 1e-4, 2000 pas), prédiction sur les 30 meilleures hypothèses (retrancher ~0,008, biais du fit sur `lg_g0`) :

| candidat (`candidates/`) | cible | CE (9 classes / autres) | retain proxy accord / KL | prédit [min, max] |
|---|---|---|---|---|
| `sub_lg_g0.pt` (référence, mesuré 0,3045) | mixcls | 3,10 ± 4,96 (8,59 / 0,35) | 0,861 / 0,047 | 0,313 [0,308–0,320] |
| contrôle (queue → mixcls, non exporté) | mixcls | 3,18 ± 5,14 | 0,854 / 0,053 | 0,312 |
| `sub_lg0_tailfitA.pt` | fitA | 2,71 ± 4,70 (6,97 / 0,56) | 0,847 / 0,059 | 0,328 [0,314–0,341] |
| **`sub_lg0_tailfitB.pt`** | fitB | 2,92 ± 4,92 (7,73 / 0,51) | 0,848 / 0,058 | **0,335 [0,323–0,347]** |

- Variantes plus longues de la queue vers `fitB` (6000 pas, lr 2e-4 à 3e-4, w-stat 0,3) : 0,339–0,346 prédits, mais retain proxy plus bas (accord 0,82–0,83) et plage plus large. La capacité de la queue sature vers 0,34 ; la version modèle complet (`fa_g1` sur JURECA) devrait faire mieux sur le retain.
- **Lecture franche** : avec s ≈ 0,58, cette famille plafonne vers 0,35. Atteindre 0,40 demanderait une information par image sur gold que les scores publics ne donnent pas.
- À faire dès le retour du SSH : rapatrier les logits `fa_g0..g3` (tâche `06_t3_fit`), les prédire avec `predict12.py`, exporter le meilleur ; supprimer `output/runs/zz_smoke` s'il existe ; tâche suivante = `fitB` en modèle complet sur plus de pas, avec et sans `--retain both`.

### 8.7 14e point et invalidation du fit hors famille (02:15)
- **Mesuré** : `sub_lg0_tailfitB` = **0,2827** public (prédit 0,335 [0,323–0,347] ; `lg_g0` 0,3045). Deuxième erreur hors famille après `lg0_bias4` (prédit 0,20–0,22, mesuré 0,2348), de signe opposé.
- Réajustement à 14 points, grille retain élargie (a jusqu'à 4, c jusqu'à 16) : RMS 0,005 → **0,013**, LOO RMSE 0,008 → **0,026** (LOO des deux points hors famille : −0,053 et +0,049). Aucune hypothèse de la famille ne place `tailfitB` sous `lg_g0` (meilleures : 0,309–0,326 pour 0,2827 mesuré). Les marginales sont plates (dcap 1 à 8 : RMS 0,013–0,014) : les paramètres gold ne sont plus identifiés.
- Diagnostic : (b) sur-ajustement à 12 runs mixcls quasi colinéaires est l'explication compatible avec les chiffres. (a) un retain réel plus sévère vu par le proxy Imagenette est rejeté : il faudrait c ≈ 14 pour expliquer −0,05 avec ΔKL = 0,011, ce qui contredit `ma_g0` (KL 0,081, public 0,289) ; un collatéral in-distribution invisible du proxy reste possible mais non testable. (c) n'est pas nécessaire : la sonde biais, pur décalage de logits identique sur les copies, est déjà mal prédite.
- **Conséquence** : `fit12.py` n'est valable que pour interpoler à l'intérieur de la recette mixcls ; il ne doit plus servir à choisir des candidats hors de cette recette. Les prédictions des §8.4 à 8.6 pour `fa_g0..g2` et `tailfitA` sont caduques (même direction que `tailfitB`, ne pas soumettre).
- **Recommandation** : garder `lg_g0` (0,3045) comme soumission finale. La recette a plafonné (4000 pas 0,3043 → 8000 pas 0,3045). Seule sonde restante non redondante : `fa_g3` (mixcls + retain CutMix in-distribution), si la tâche 06 a tourné et si le SSH revient ; gain incertain.

### 8.9 Le goulot est le retain sur des images de la même distribution (03:40 → 05:00, dim. 22 nov.)
**Constat (mesuré)**
- Outils :
  - `gen_retain.py` génère 900 images SD-Turbo des 45 classes du forget set (noms tirés de `forensics/class_names.json`), en local sur MPS ; le style colle à celui du dataset ;
  - `probe_gen.py` compare chaque modèle à `orig` sur 225 de ces images, tenues à part ;
  - sur cette validation, `orig` est d'accord avec la classe visée dans 40 % des cas.
- Les 14 modèles déjà soumis s'éloignent tous fortement d'`orig` sur ces images : accord avec orig 0,74–0,84, KL(orig‖sub) 0,10–0,46.
  - `lg_g0` : accord 0,77, KL 0,30.
  - Les deux sondes ratées sont celles qui abîment le plus ces images : `lg0_bias4` (KL 0,46) et `lg0_tailfitB` (KL 0,41).
  - Sur Imagenette, `lg0_tailfitB` ne bougeait que de +0,011 en KL, ce qui explique que le fit ne l'ait pas prévu.

**Recalibration (`fit_indist.py`, `predict_gen.py`)** : même famille de gold que `fit12.py`, retain = accord^a / (1 + c·KL) mesuré sur ces images.
- Erreur en leave-one-out sur les 14 points : 0,0148, contre 0,0195 avec Imagenette.
- Échelle inexpliquée s : 0,90, contre 0,57.
- Erreurs sur les deux sondes : −0,024 et +0,028, contre −0,038 et +0,049.

**Correctif** : `unlearn.py --retain-val` ; retain = Imagenette ou DiffusionDB + images générées (×4 à ×8, cible `orig`) ; départ à chaud depuis `lg_g0`, +3000 pas.
- L'écart avec `orig` sur ces images est divisé par 5 : KL 0,30 → 0,05–0,06, accord 0,77 → 0,91–0,92.
- L'oubli est inchangé : CE 3,1 ± 4,9, accuracy 0,62–0,64.

| candidat (`candidates/`) | config | public prédit [min–max] | accord / KL (images générées) | accord / KL (Imagenette) |
|---|---|---|---|---|
| `sub_gm_g2.pt` | α=10, DiffusionDB + images générées ×4 | 0,438 [0,35–0,53] | 0,920 / 0,063 | 0,876 / 0,044 |
| `sub_gn_g1.pt` | α=10, Imagenette + images générées ×8 | 0,434 [0,35–0,53] | 0,911 / 0,060 | 0,868 / 0,040 |
| `sub_gm_g0.pt` | α=30, Imagenette + images générées ×8 | 0,416 [0,33–0,51] | 0,920 / 0,050 | 0,872 / 0,035 |

- Les prédictions sont **supposées** : la zone KL ≈ 0,05 n'a jamais été mesurée en public, d'où la largeur des intervalles.
- Sonde décisive : `sub_gm_g2.pt`.
- **Résultat (04:52)** : `sub_gm_g2` = **0,3128** public (prédit 0,438). C'est notre meilleur score, mais seulement +0,008.
  - Refit sur 15 points : le terme de retain sur images générées devient quasi nul (a = 0, c = 0,25–0,5) ; s = 0,57 ; RMS 0,0136.
  - L'hypothèse « le retain sur la même distribution est le goulot » est largement réfutée.
  - gn_g1 est prédit à 0,324 et gm_g0 à 0,311 : rien qui justifie une autre sonde.
  - **Soumission finale T3 recommandée : `sub_gm_g2.pt`.**
- **Sondes de directions radicalement différentes (05:00–06:00)**, score public mesuré :

| sonde | ce qui change | public |
|---|---|---|
| `fc_g1_oof9` | professeur hors pli : oubli fort sur les 36 classes partielles (acc 0,69) | **0,105** |
| `fb_g2_sis` | 9 classes redirigées vers leurs classes sœurs | **0,185** |
| `gm_g0` | α=30 au lieu de 10 | 0,3130 (≈ `gm_g2` 0,3128) |

- Conclusion :
  - toute modification de la distribution « fausse » ou de l'oubli des classes partielles fait chuter le score ;
  - α est sans effet entre 10 et 30 ;
  - la famille mixcls plafonne à ~0,31.
- Runs de la même famille, tous à oubli CE 3,1 ± 4,95, acc 0,62 et retain (images générées) 0,89–0,92 / KL 0,05–0,07 :
  - `l2` : 6000 pas, lr 1e-4, α 5, seed ;
  - `au2`, `au3` : altérations larges ;
  - `ws2` : w-stat / α ;
  - soupe `soupG4`.
  - Aucun ne se détache : prédits 0,33–0,34, avec un fit dont le RMS est de 0,02.
- Variantes `split` / `--pop-stat` (bp, ps, od) : elles appuient sur les classes partielles, avec une acc forget de 0,54–0,60 ; prédites 0,00–0,27. Écartées.
- Équipe : meilleur score 0,3158 (session Modal d'Amil : recette `lg_g0` à 4000 pas + retain text-to-image). Le monde jouet à vrai gold (13k images générées, 4 gold') est la seule source d'information manquante sur le comportement de gold.
- Incidents :
  - le quota d'inodes du scratch du projet est plein, donc les runs écrivent dans `/p/project1/training2666/geran1/t3runs/` ;
  - les jobs hors réservation ont été annulés (sanction JSC), avant l'écriture des `sub_*.pt` ; ceux-ci ont été refaits en local.

### 8.8 Nouvelles familles (02:20 →) : erase + repair, professeur hors pli, retain DiffusionDB, altérations
Contexte : leaderboard à 02:21 = 0,41 / 0,41 / 0,39 / 0,38 / 0,35 / 0,35 / 0,34, nous 9e à 0,3045. On ne choisit plus par prédiction du fit : chaque idée est une sonde au leaderboard (seule la meilleure soumission compte).

- **Origine d'`orig` (mesuré, `ident_pre.py`)** : similarité cosinus des poids avec augreg in21k, augreg2, deit3 (in1k, in22k), MAE = 0,00. LayerScale est passé de 1e-4 à ~1e-2. `orig` est entraîné depuis zéro (ou depuis un pré-entraîné privé). Un k-fold à partir d'un pré-entraîné public n'est donc pas « la même famille » ; on part d'`orig` avec les derniers blocs réinitialisés.
- **Tâche 06 (`fa`)** : 3 runs sur 4 ont planté au premier checkpoint sur « Disk quota exceeded » (inodes). Seul `fa_g0` (cible fitA) existe ; non recommandé (même direction que `tailfitB`, mesuré 0,2827). `fa_g3` (retain CutMix) n'existe pas.
- **Copies altérées (mesuré, `alt_check.py`, 500 images)** : l'oubli de `lg_g0` se transfère aux altérations légères (CE propre 2,96 ; flip 2,67 ; redimensionnement ×0,5 2,89 ; quantification 2,83 ; bruit 0,05 2,93 ; flou 2,87 ; recadrage léger 2,55) alors qu'`orig` y reste à CE ≈ 0. Pas de trou structurel « copies restées mémorisées ». `orig` s'effondre en niveaux de gris (CE 6,5, acc 0,29) et sous bruit fort (CE 0,8).
- **Sonde biais (§8.5), lecture corrigée** : 0,2348 n'écarte pas que gold ne prédise jamais les 9 classes sur les images forget ; un biais global casse aussi les images de contrôle de ces classes (terme retain). Ce qui est établi : un oubli par classe perd 0,07 ; il faut un oubli par instance.
- **Code ajouté** : `unlearn.py --method er --reinit <blocs> [--reinit-mlp]` (erase + repair : réinitialisation des derniers blocs + tête, réparation par KL vers `orig` sur données de substitution uniquement), `--retain-file` (liste de fichiers, dont `retain_ddb_u8.npy` = 3 000 images DiffusionDB converties par `prep_ddb.py`), `--faug wide` (redimensionnement, quantification, bruit, flou, couleur, recadrages), `--save-last` ; `kfold_oof.py` (professeurs hors pli stratifiés, redistillation d'`orig`, variante `--drop-capped` avec colonnes des 9 classes masquées, q_i moyenné sur 4 vues légères).
- **Tâche `08_t3_er`** (créneau 03:12, 3000 pas) : g0 erase+repair MLP des blocs 9–11 ; g1 erase+repair blocs 10–11 entiers ; g2 recette `lg_g0` + altérations larges ; g3 distillation vers la cible hors pli DINOv2 sans w-stat.
- **Tâche `09_t3_oof`** (~40 min, worker suivant) : professeurs hors pli dans la vraie architecture puis distillation modèle complet 6000 pas sans w-stat (variante standard et variante « 9 classes »), plus les deux meilleures sondes de 08 prolongées.
- Le k-fold sur le GPU du login (RTX 8000 partagée, bfloat16 émulé) n'a pas rendu un pli en 13 min : arrêté.

### 8.9 Résultats des tâches 08 et 09, banc à vrai gold (03:10–04:35)
**Tâche `08_t3_er`** (mesuré, proxy) :
- erase + repair (blocs 6–11 ou 8–11 réinitialisés + tête, réparation sur Imagenette + DiffusionDB + CutMix) : accord retain proxy 0,48 / 0,52, KL 0,75 / 0,59. Passerait très probablement sous la porte des 70 %. Piste fermée.
- Mesure locale associée (`tail_er.py`) : réinitialiser le MLP du bloc 11 + la tête puis réparer sur Imagenette seul redonne acc 0,993 sur le forget set. La mémorisation est dans les features profondes, pas dans la tête.
- `fb_g2` cible « sœurs » (85 % de la masse des images des 9 classes sur les sœurs non plafonnées, reste = mixcls, départ `lg_g0`, checkpoint du pas 2500) : CE 3,47 ± 4,84, retain proxy 0,859 / 0,052. Candidat `candidates/sub_fb_g2_sis.pt`.
- `fb_g3` distillation vers la cible hors pli DINOv2 sans w-stat : retour au profil « medium » (acc forget 0,91, CE 0,27). Non recommandé.

**Tâche `09_t3_oof`** (mesuré) :
- Professeur hors pli standard `oofA` (4 plis, blocs 8–11 + tête réinitialisés, 600 pas) : acc hors pli 0,37 (sous-entraîné) ; garde-fou déclenché, `fc_g0` retombé sur la cible DINOv2 (profil « medium », non candidat).
- Professeur sans aucune image des 9 classes `oof9` : acc 0,48 sur les 36 autres classes, 0,000 sur les 9. Distillation modèle complet 6000 pas sans w-stat → `fc_g1` : CE 2,78 ± 3,00 (9 classes 6,53 / acc 0,002 ; autres 0,91 / acc 0,69), retain proxy 0,858 / 0,065. Sans y être forcé, le modèle envoie les images des 9 classes vers leurs sœurs (6,7,8→5 ; 29,30,31,33→32 ; 38→36,35,37 ; 0→2,3). Candidat `candidates/sub_fc_g1_oof9.pt`.

**Banc à vrai gold (`bench.py`)** : monde = les 3 500 images (pseudo-labels) ; contrôle 531 (15 % stratifié), forget' 860 (classes 0, 7, 30 entières + 20 % des autres), retain' 2109. orig' et 2 gold' = ViT-B appris de zéro, 3000 pas.
- orig' mémorise (forget' acc 1,000) mais ne fait que 0,205 sur le contrôle ; gold' 0,18 / 0,17. **Régime bien plus faible que la vraie tâche : lire le sens des effets, pas les valeurs.**
- gold' sur forget' : CE 8,42 ± 5,01, bimodal (quantiles 10/50/90 % : 0,86 / 9,5 / 14,2). Classes entièrement retirées : acc 0,000, masse 0,004, CE 12,6, prédictions vers les sœurs. Autres classes : acc 0,21, CE 5,8.

| méthode appliquée à orig' | score | forget (ms, ps) | retain réel in-distribution (ratio acc, KL) | retain proxy Imagenette (accord, KL) |
|---|---|---|---|---|
| gold' lui-même | 0,33 / 0,32 | 0,90 (0,99, 0,91) | 0,37 / 0,35 (0,88, 1,39) | — |
| recette lg (mixcls, 2000 pas) | 0,168 | 0,242 (0,45, 0,54) | 0,695 (0,972, 0,399) | 0,850 (0,872, 0,026) |
| lg + retain CutMix/DiffusionDB | 0,160 | 0,217 (0,43, 0,51) | 0,740 (0,954, 0,290) | 0,853 (0,875, 0,025) |
| lg court (400 pas) | 0,162 | 0,235 (0,44, 0,54) | 0,689 (0,991, 0,437) | 0,821 (0,854, 0,041) |
| erase + repair | 0,057 | 0,086 (0,15, 0,59) | 0,661 (1,000, 0,513) | 0,361 (0,474, 0,315) |
| oracle (distillation vers gold' réel, sans w-stat) | 0,251 | 0,420 (0,48, 0,88) | 0,597 (0,890, 0,490) | 0,764 (0,802, 0,050) |

Lectures :
1. **Dégât collatéral invisible confirmé** : la KL réelle sur le contrôle in-distribution est ~15 fois celle que voit le proxy Imagenette (0,40 contre 0,026). Le retain de substitution CutMix + DiffusionDB la réduit à 0,29.
2. **Même avec la cible parfaite, sans terme de statistiques, ms plafonne à 0,48** (CE 4,2 ± 2,8 au lieu de 8,4 ± 5,0) : la moyenne d'ensemble est plus douce qu'un gold individuel. Il faut un terme moyenne/écart-type au niveau d'un modèle.
3. Classes entièrement retirées → sœurs.
4. Le nombre de pas compte peu sur le banc ; erase + repair est mauvais.
- Runs « professeur » du banc absents (professeurs trop faibles sur 645 images). Le premier passage de l'évaluation a planté (autocast imbriqué), corrigé.

**Tâche `11_t3_sis`** (en file, 2800 pas depuis `lg_g0`, statistiques par population `--pop-stat`) : lignes des 9 classes = répartition du professeur `oof9`, reste = mixcls ; g0 CE visée 9,7 sur les 9 classes ; g1 12,7 ; g2 = g0 + retain CutMix/DiffusionDB ; g3 = g2 avec α = 10. Écritures de checkpoint rendues non fatales.

### 8.10 Voie C : modifications sans entraînement de `soup_C` (dim. 22 nov., 08:15–08:50) — mesuré
Code : `tasks/unlearning/solution_c/` (`c_feats.py` met en cache les features avant la tête, donc tout réglage de tête est exact et instantané ; `c_head.py` température / biais ; `c_lam.py` interpolation de poids ; `c_verify.py` validation serveur + forward complet ; `c_public.py` table des modèles soumis). Tout en local (MPS) : les workers `hqc` n'ont jamais démarré (PENDING, BadConstraints).

Base `soup_C` (0,3166). Accord / KL à T=2 avec `orig` sur 225 images générées tenues à part ; « p part. » = p(vraie classe) à T=2 sur les images des classes partielles.

| candidat | CE oubli | 9 classes CE / acc | partielles CE / acc | p part. | accord / KL | public |
|---|---|---|---|---|---|---|
| `soup_C` | 3,11 ± 4,94 | 8,63 / 0,06 | 0,33 / 0,91 | 0,62 | 0,920 / 0,057 | 0,3166 |
| température 0,85 (tête × 0,85) | 2,71 ± 4,22 | 7,45 / 0,06 | 0,34 / 0,91 | 0,50 | 0,920 / 0,105 | **0,2789** |
| température 1,15 | 3,52 ± 5,66 | 9,84 / 0,06 | 0,35 / 0,91 | 0,70 | 0,920 / 0,061 | **0,3098** |
| λ = 0,85 (θ = orig + λ(soup − orig)) | 2,33 ± 4,14 | 6,64 / 0,14 | 0,17 / 0,95 | 0,76 | 0,929 / 0,040 | non soumis |
| λ = 1,10 | 3,48 ± 5,21 | 9,46 / 0,04 | 0,49 / 0,87 | 0,51 | 0,902 / 0,071 | pas de fichier |
| λ = 1,15 | 3,64 ± 5,27 | 9,74 / 0,03 | 0,58 / 0,84 | 0,46 | 0,898 / 0,079 | en attente |
| λ = 1,25 | 3,89 ± 5,29 | 10,05 / 0,01 | 0,80 / 0,77 | 0,37 | 0,889 / 0,096 | pas de fichier |

- **Température : close.** Elle ne change aucune prédiction (accuracy de contrôle identique). Parabole sur les 3 points : optimum vers 1,05, gain ≤ 0,003. À 0,85 la KL retain explique environ −0,014 des −0,038 ; à 1,15 elle ne coûte qu'environ −0,001, donc les −0,007 viennent du terme d'oubli.
- **Biais des 9 classes : clos** par deux points publics antérieurs (−4 : −0,070 ; +1,5 : −0,007) : optimum à +0,2, gain nul.
- Température par lignes (9 classes seules, ou 36 classes partielles seules) : change les prédictions, accord 0,82–0,88 pour un facteur 1,15–1,3. Écarté.
- **λ est le seul bouton sans entraînement qui agit sur les images partielles** (acc 0,91 → 0,84 à λ = 1,15) ; coût retain mesuré : accord −2,2 points, KL +0,022.
- Table des 14 modèles soumis avec logits locaux (`c_public.py`) : `m3_g2` / `m3_g3` (0,205 / 0,208) ont le même niveau de CE que la famille `lg` (0,30) mais sont nets sur les images partielles (p part. 0,80–0,82 contre 0,60). La douceur à T=2 sur les images partielles est donc fortement notée ; `tailfitB` (0,44, public 0,283) et la température 0,85 (0,50, 0,279) montrent qu'on ne gagne pas en adoucissant davantage par la tête.
- Supposé (non mesuré) : l'asymétrie des deux points de température (durcir coûte 4 fois moins qu'adoucir, hors retain) est compatible avec une CE gold par modèle plus haute que la nôtre, pas avec un écart-type gold de 4,0.

## 9. Dimanche 22 nov., 07:45 → : ré-entraînement approché (échec au banc) puis sondes sur le niveau de CE

### 9.1 Ré-entraînement approché (`retrain.py`) — mesuré, perd nettement
Méthode : réinitialiser la tête + les K derniers blocs d'`orig` (LayerScale gardé), puis les ré-apprendre uniquement par distillation d'`orig` (KL à T=2 et T=1 + CE sur l'argmax) sur Imagenette + DiffusionDB + images générées, augmentations fortes + CutMix, 2000 pas. Aucune image du forget set, aucun terme d'oubli. Variante douce : shrink-and-perturb 50 % sur 6 blocs.

| variante | banc (score exact, orig' + gold') | vrai `model.pt` : CE forget / acc | accord avec orig, images générées |
|---|---|---|---|
| recette mixcls `lg` (référence) | 0,168 | 3,1 ± 4,9 / 0,62 | 0,92 |
| K = 1 | 0,008 | 0,02 ± 0,12 / 0,993 | 0,876 |
| K = 2 | 0,024 | 0,32 ± 0,74 / 0,90 | 0,738 |
| K = 4 | 0,052 | 1,89 ± 1,84 / 0,44 | 0,627 |
| shrink-and-perturb, 6 blocs | 0,003 | 0,35 ± 0,72 / 0,87 | 0,756 |

Lecture : la réparation ré-apprend la fonction d'`orig`, mémorisation comprise (les blocs bas la portent). Dès qu'on efface assez pour oublier, l'accord tombe sous 0,75 (risque de porte à 70 %). Sans vrai retain set, les recettes « erase then repair » ne s'appliquent pas. Piste fermée. Limite du banc : orig' ne fait que 0,205 sur son contrôle ; on lit le sens, pas les valeurs.

### 9.2 Sondes sur le niveau de CE (`unlearn.py --stat-target`, `--pop-target`, `--reservoir`)
Seul axe jamais testé en public : la moyenne et l'écart-type de la CE sur le forget set (toute la famille était à 3,1 ± 4,9, imposé par `w-stat`). Départ `soup_C` (0,3166), mixcls, retain DiffusionDB + images générées. `--reservoir` calcule moyenne/écart-type sur les 3 500 images (tampon) au lieu du batch : la cible est atteinte à ±0,25. Non validable au banc (le niveau de CE du banc n'est pas celui du vrai gold) : ce sont des sondes leaderboard. Environ 200 runs (5 grilles de 40 sur les nœuds hors réservation), tous valides, accord images générées 0,86–0,93. Outil : `profile_subs.py`.

Scores publics mesurés (soumis par la session principale) :

| fichier (`candidates/`) | CE forget (propre) | 9 classes / partielles (acc) | accord / KL images générées | public |
|---|---|---|---|---|
| `sub_soup_C_uniform` (base) | 3,11 ± 4,94 | 8,63 (0,06) / 0,33 (0,91) | 0,920 / 0,057 | 0,3166 |
| `soup_C` × 0,85 (température, voie C) | 2,71 ± 4,22 | — | — | 0,2789 |
| `soup_C` × 1,15 (voie C) | 3,52 ± 5,66 | — | — | 0,3098 |
| `sub_ceg_g_m3.8_s3.5` | 3,73 ± 3,83 | 8,14 (0,01) / 1,52 (0,63) | 0,907 / 0,067 | 0,3159 |
| `sub_pol_C_a20_s1` | 5,63 ± 4,39 | 10,47 (0,00) / 3,20 (0,39) | 0,924 / 0,065 | **0,3209** |

Lecture : la réponse au niveau de CE est presque plate entre 3,1 et 5,6 de moyenne. Passer l'accuracy des classes partielles de 0,91 à 0,39 ne coûte rien et ne rapporte presque rien.

### 8.10 Voie B : cibles par image apprises ou dérivées (dim. 22 nov., 08:15–09:40)
Code : `solution/unlearn_b.py` (= `unlearn.py` + `--target-file`, `--holdout`, `--cap-boost`, `--part-boost`, `--anchor`, `--freeze-head`, `--freeze-last`), `solution/collat.py` (accord avec `orig` par groupe de classes), `solution/make_targets_b.py` (fichiers de cibles). Runs : `/p/project1/training2666/geran1/t3runs/b_runs` et `b_bench`.

**Résultat public (mesuré)** : `sub_b_csx_soup.pt` = **0,2818** (base `soup_C` 0,3166 ; meilleur 0,3209).
- Recette : départ `soup_C` + 2500 pas, cible « csx » = moyenne d'`orig` sous augmentations fortes et très fortes, vraie classe retirée sur les 9 classes ; w-stat 0,3 visant 2,9 ± 4,9 ; α 20 ; soupe de 2 graines.
- Profil : CE 3,24 ± 5,02 ; 9 classes 8,93 / acc 0,06 ; partielles 0,38 / acc 0,92 ; accord images générées 0,916 (KL 0,070).
- Le banc à vrai gold' donnait cette cible gagnante (0,183 contre 0,158 pour mixcls). **Le banc n'est donc pas fiable pour la forme de la cible.**
- Lecture : retirer toute la masse de la vraie classe sur les 9 classes est pénalisé (cohérent avec « 9 colonnes éteintes » 0,068 et « classe sœur fixe » 0,185).

**Cible apprise sur le banc (idée fermée, mesuré hors ligne sur les 860 images forget' du banc)**
- Les signaux d'orig' prédisent mal le gold' par image sur les classes partielles : Spearman avec p(vraie) de gold' = 0,21 (augmentation forte), 0,20 (très forte), 0,18 (moyenne), −0,03 (marge sur image propre). La classe explique 31 % de la variance.
- Plafond du terme KL selon la forme de la cible (1 − KL(gold'‖cible)/KL(gold'‖orig')) : mixcls 0,49 ; cls:strong 0,49 ; csx 0,55 ; mixcls adouci (puissance 1/1,5) 0,56. Cible bimodale « images les plus dures » sur les partielles : 0,35–0,39 contre 0,39 pour mixcls.

**Oubli par classe ou par instance (mesuré, run depuis `orig`, 20 % des images à oublier tenues hors de l'entraînement)**
- Images des 9 classes jamais utilisées comme images à oublier : acc 0,57, CE 2,2 (entraînées : acc 0,20, CE 7,0). Classes partielles tenues à part : acc 0,96.
- L'oubli des 9 classes déborde donc sur des images non entraînées des mêmes classes ; le coût sur le contrôle caché n'est pas mesurable ici.
- Correctifs testés (acc des images tenues à part des 9 classes / accord images générées) : sur-échantillonnage du retain prédit dans les 9 classes 0,56 / 0,876 ; mosaïques d'ancrage 25 % 0,64 / 0,876 ; ancrage 50 % sur les 9 classes 0,71 / 0,867 ; tête gelée 0,585 / 0,880 ; tête + 4 derniers blocs gelés 0,51 / 0,884. Aucun ne garde l'accord ≥ 0,90.

**Banc à vrai gold' (2000 pas, sens seulement)**

| variante | score | oubli (ms, ps) | retain réel (ratio acc, KL) |
|---|---|---|---|
| recette mixcls (`unlearn_b.py`) | 0,158 | 0,228 (0,44, 0,52) | 0,695 (0,954, 0,373) |
| sur-échantillonnage retain | 0,150 | 0,228 | 0,657 |
| ancrage 25 % / 50 % | 0,149 / 0,149 | 0,212 / 0,207 | 0,706 / 0,721 |
| tête gelée / + 4 blocs | 0,161 / 0,132 | 0,227 / 0,201 | 0,707 / 0,657 |
| cible adoucie 1/1,5 | 0,162 | 0,271 | 0,599 |
| cible csx | 0,183 | 0,290 (0,48, 0,60) | 0,631 |
| oubli partiel par image (36 % / 57 % des partielles) | 0,192 / 0,202 | 0,324 / 0,357 | 0,591 / 0,567 |
| mixcls + CE par population (voie A) | 0,190 | 0,312 | 0,610 |

- Dans le banc, la cible par image d'oubli partiel ne fait pas mieux que mixcls avec une CE visée par population.

**Runs réels non soumis (départ `soup_C`, 2500 pas)** : cibles adoucie, pondérée par la part de classe retirée : profil de la famille (CE 3,1–3,2 ± 5,1 ; accord 0,90–0,92). Oubli partiel par image, soupe de 5 graines (`b_runs/soup_pw5.pt`) : CE 3,98 ± 4,96 ; 9 classes acc 0,02 ; partielles CE 1,26 / acc 0,77 ; accord 0,902.
