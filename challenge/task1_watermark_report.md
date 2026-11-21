# Task 1 — Localisation de watermark : rapport (état au sam. 21 nov. ~19:00)

## Ce qui a été essayé (agent précédent, `solution/`)
- `features.py` + `baseline.py` : décision globale (accord moyen tanh(logit)·(2·bit−1) avec le message connu → mode « presque tout marqué » ou « presque propre »), puis seuillage des per_position_logits (PixelSeal, TrustMark). Seuils réglés sur le fold train, évalués sur le fold held-out (`common.split_train` : 25 images par groupe (méthode, full) → 150 images).
- `synth.py` + `train_seg.py` : U-Net entraîné sur des paires synthétiques (COCO + masques aléatoires) et sur le fold train réel. Il plafonne à 0.455 en val, soit le niveau de la décision de mode seule.
- explore5–9, `sliding.py`, `blocktest.py` : sliding windows, attribution par gradient, ré-embedding. Les AUC pixel restent faibles (0.50–0.63).

## Chiffres held-out (mIoU officiel, fold val de 150 images)
| méthode | all-zeros | mode seul (= oracle de mode) | baseline (mode + ppl) |
|---|---|---|---|
| pixelseal | 0.249 | 0.454 | 0.479 |
| trustmark | 0.260 | 0.446 | 0.459 |
| stegastamp | 0.250 | 0.460 | 0.460 |
| **global** | 0.25 | 0.453 | **0.466** |

Soumission baseline déjà écrite : `tasks/Hackathon_Watermark_Localisation_Task/solution/submission_baseline.pt`. Elle correspond au val 0.466 et n'a pas été soumise.

## Découverte clé (nouvelle)
- Environ la moitié des images « partielles » (129 sur 255) ont un masque exactement rectangulaire. Le jeu de données a été fabriqué ainsi : **crop → embed(crop, message) → recollage**.
- Si on décode exactement ce crop, on retrouve le message connu avec une bit accuracy de **0.9–1.0**, contre ~0.5 pour l'image entière ou un crop éloigné (`solution/croptest.py`, `solution/basin.py`).
- Tolérance au décalage (`basin.py`) :
  - TrustMark : environ ±8–16 px ;
  - PixelSeal et StegaStamp : environ ±2–4 px ;
  - agrandir la bbox GT de 2–4 px aide souvent.
- Les masques partiels irréguliers ne sont pas des rectangles tournés (`rottest.py`, résultat négatif). C'est probablement un embed de l'image entière, puis un collage par blob : cas difficile, laissé au baseline.

## Méthode en cours : recherche de rectangle
- Code : `solution/rectsearch.py`.
- Grille grossière de boîtes (taille ≥ 24 px, ratio d'aspect ≤ 2, environ 63k boîtes), score = accord doux avec le message.
- Crops extraits et redimensionnés en batch par `roi_align`, puis top-16 avec NMS, puis descente par coordonnées sur les 4 bords (pas 16 → 1 px).
- Lancement : `solution/run_search.py`, cache dans `solution/cache/search_{split}_{method}{tag}.npz`. Les images classées « full » sont sautées.
- Combinaison et évaluation : `solution/rect_eval.py`. En mode propre, si score > t_rect, on prédit la boîte agrandie de d px, sinon le baseline. t_rect et d sont réglés par méthode sur le fold train, puis rapportés sur val. Avec `--test out.pt`, le script écrit la soumission.
- Smoke test local OK (un rectangle TrustMark retrouvé avec un score de 1.000).
- Gain attendu : environ +0.4 de mIoU sur environ 25 % des images, soit environ +0.1 au global si la recherche les retrouve.

## Comment relancer sur JURECA
- Tâches mises dans la file partagée `/p/scratch/training2666/geran1/IncomingPHd/queue/pending/` :
  - `10_t1_rect_train.sh` (SPLITS=train) ;
  - `11_t1_rect_test.sh` (SPLITS=test TAG=_t) ;
  - `30_t1_rect_fine_train.sh` (grille fine 16,1.25,0.2,2.0, TAG=_fine).
- Elles appellent `bash solution/job_rect.sh` (GPU 0 TrustMark, GPU 1 PixelSeal, GPU 2–3 StegaStamp). Logs : `Hackathon_Watermark_Localisation_Task/output/rect*.log`.
- Ensuite :
  1. rapatrier `solution/cache/search_*.npz` en local (`rsync -az -e ssh juelich:.../solution/cache/ solution/cache/`) ;
  2. lancer `python solution/rect_eval.py --test solution/submission_rect.pt` (option `--data release` si exécuté sur JURECA).
- Attention : `rect_eval.py` charge tous les `search_{split}_{m}*.npz`. Exclure `_fine` du réglage si on compare les grilles (filtre à ajouter).

## Prochaines étapes
1. Récupérer les résultats train, régler t_rect et d, lire les chiffres held-out, puis écrire `submission_rect.pt` dès que le test est fini.
2. Si le débit le permet : grille plus fine pour PixelSeal et StegaStamp (bassin étroit), et des boîtes plus petites (< 24 px).
3. Mode « full » (zone propre restaurée, irrégulière) : piste explore8 (compléter le watermark sur une région candidate C et mesurer la hausse d'accord).
