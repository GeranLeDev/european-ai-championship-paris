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

**Candidats rapatriés et validés en local** (chargement strict, float32, valeurs finies, 328 Mo ; NON soumis), dans `tasks/unlearning/candidates/` :
1. `sub_m1_g2.pt` : mixcls, α=5. Le plus haut sous toutes les hypothèses ajustées. Risque : retain proxy plus bas (KL 0,055, accord 0,858) que `r2_g2` (0,039 / 0,892).
2. `sub_m1_g0.pt` : mixcls, w-stat 0,05. Prédit 0,254–0,274, donc équivalent à `r2_g2` ; retain proxy meilleur (KL 0,033, accord 0,905).
3. `sub_m3_g3.pt` : cls pur, α=20, meilleur retain (KL 0,024, accord 0,918). Prédit 0,19–0,25 : sonde pour mesurer l'effet de la suppression seule, pas un candidat au record.

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
