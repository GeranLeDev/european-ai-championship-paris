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

## Recherche de rectangle : résultat (sam. 21 nov. 21:30)

**Fichier prêt : `tasks/Hackathon_Watermark_Localisation_Task/solution/submission_rect.pt`** (même format et même ordre d'images que `submission_baseline.pt` : `image_names` list[str] de 1020, `masks` bool [1020,128,128]). **Non soumis** (accord utilisateur requis). 131 images changent par rapport au baseline (81 TrustMark, 50 PixelSeal, 0 StegaStamp).

### Held-out mesuré (fold val de 150 images, t_rect et d réglés sur le fold train, mIoU officiel)
| méthode | baseline | baseline + rectangle | t_rect, d | commentaire |
|---|---|---|---|---|
| pixelseal | 0.479 | **0.528** | 0.40, 3 | 7 rectangles sur 15 retrouvés en val |
| trustmark | 0.459 | **0.573** | 0.60, 0 | 13 rectangles sur 14 retrouvés, mIoU 0.87 sur ces images |
| stegastamp | 0.460 | 0.460 | (non utilisé) | la recherche échoue, baseline conservé |
| **global** | **0.466** | **0.520** | | gain +0.054, IC bootstrap 95 % [+0.030 ; +0.081] |

- Le gain ne dépend pas du réglage fin : toutes les cases (t_rect, d) testées en val battent le baseline pour PixelSeal et TrustMark.
- Le baseline local 0.466 a donné 0.4584 en public : on s'attend donc à ~0.51 en public (supposé, non mesuré).
- Le gain attendu initial (+0.1) n'est pas atteint. Les rectangles font bien 25 % des images (129/510 au train), mais seul TrustMark les retrouve presque tous.

### Ce qui limite (mesuré)
- **PixelSeal** : les rectangles fins (un côté < ~20-30 px) ne donnent aucun accord, même en décodant exactement la boîte vraie (`basin.py` : 7 cas sur 14 à ~0.00). Aucune grille plus fine ni boîte < 24 px ne les retrouvera avec ce score. Les rectangles moyens donnent 0.13-0.28 sur la boîte exacte, au niveau du bruit (0.11-0.20).
- **StegaStamp** : plancher de bruit du score à 0.30-0.46, 1 rectangle retrouvé sur 23 dans les 44 images train cherchées. La recherche test n'a pas été lancée (58 s/image, soit ~45 min de nœud pour un gain mesuré nul).
- Les masques 128 ont des coins arrondis à 0 : compter les rectangles sur les masques 512 (corrigé dans `rect_eval.py`).
- Grille fine (`30_t1_rect_fine_train`, 304k boîtes, 5x plus chère) : rangée dans `queue/parked_t1`, non lancée, car ni PixelSeal ni StegaStamp ne manquent de densité de grille.

### Code et caches
- `solution/rectsearch.py` : score d'une boîte = accord doux avec le message, crops par `roi_align`, top-16 + NMS, descente par coordonnées. StegaStamp utilise un graphe ONNX réduit au décodeur (`stega_stamp_dec.onnx`, 4x plus rapide, sorties identiques ; le graphe complet faisait un OOM à bs=256).
- `solution/run_search.py` : reprise sur cache (`--skip-dir` + fichier de sortie relu), sauvegarde atomique après chaque image, repli de batch en cas d'OOM.
- `solution/rect_eval.py` : sélection des caches par tag exact (`--train-tags ",_r,_r2" --test-tags "_t,_t_r,_t_r2,_t_r3"`, les caches `_fine`/`_dbg` ne fuient plus), détail par type de masque, IC bootstrap, `--test out.pt`.
- Tâches JURECA : `job_rect.sh` (première version, trop lente : PixelSeal 56 s/image), `job_rect2.sh`/`job_rect3.sh` (version réduite sur 4 GPU, grille 24,1.4,0.33,2.0 pour PixelSeal), `job_rect4.sh` (rattrapage après l'incident de quota disque de 20:46).
- Débits mesurés sur A100 : TrustMark 18 s/image (grille 63k), PixelSeal 27 s/image (grille 30k), StegaStamp 58 s/image (grille 30k).
- Caches `solution/cache/search_{train,test}_*.npz` complets pour TrustMark et PixelSeal (toutes les images en mode « propre »).

Pour régénérer : `python solution/rect_eval.py --train-tags ",_r,_r2" --test-tags "_t,_t_r,_t_r2,_t_r3" --test solution/submission_rect.pt`.

## Prochaines étapes
1. Soumettre `submission_rect.pt` si l'utilisateur donne son accord.
2. Les 75 % restants plafonnent à ~0.46-0.50 : masques irréguliers (partiels) et mode « full » avec zone propre restaurée. C'est là qu'est la marge, pas dans une grille plus fine.
3. PixelSeal, rectangles fins : chercher un autre signal (le watermark y a peut-être été posé autrement que par crop → embed → recollage).
