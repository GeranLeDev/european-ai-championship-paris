# TASK 3 — Unlearning ViT-B/16 : rapport (sam. 21 nov., soir)

Code : `tasks/unlearning/solution/` (local) = `/p/scratch/training2666/geran1/IncomingPHd/unlearning/solution/` (JURECA).
Rien n'a été soumis (pas d'appel à `submission.py`) et rien n'a été commité.

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
- Pour récapituler et exporter :
  ```bash
  cd /p/scratch/training2666/geran1/IncomingPHd/unlearning && source .venv/bin/activate
  python solution/collect.py
  cd solution && python make_submission.py ../output/runs/<run>/best.pt ../output/sub_<run>.pt
  ```
  `collect.py` donne, par run, la config et le meilleur score proxy sous chaque hypothèse.

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
