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

## Banc d'assemblage des deux pipelines (dim. 22 nov. ~01:30)

Code : `tasks/Hackathon_Watermark_Localisation_Task/solution/ensemble.py`, test : `solution/test_ensemble.py` (4 tests, ~15 s, passent en local). Rien n'est soumis par ce script.

### Format d'échange
Un fichier par (pipeline, outil, split) dans un même dossier : `<pipeline>_<outil>_<split>.npz` avec `names`, `probs` float16 [N,128,128] et, pour `train`, `used_in_training` bool (absent = on suppose que les noms pairs ont servi). `train_seg2.py --probs-dir` écrit déjà ce format.

### Commandes
```bash
cd tasks/Hackathon_Watermark_Localisation_Task
.venv/bin/python solution/test_ensemble.py
# Amil, depuis son dossier modal_t1 (numpy + scipy suffisent) :
python <repo>/tasks/Hackathon_Watermark_Localisation_Task/solution/ensemble.py import-amil \
  --run pixelseal=pixelseal_research/runs/hires_c+pixelseal_research/runs/frozen_a,trustmark=runs/trustmark_mix,stegastamp=runs/stegastamp_mix \
  --out probs_export
# assemblage + soumission (non envoyée) :
.venv/bin/python solution/ensemble.py eval --probs <dossier> --pipelines amil,seg2 --pp thr,K \
  --fallback solution/submission_rect.pt --report output/ensemble_report.json --test solution/submission_ens.pt
```

### Ce que fait `eval`
- Validation propre par outil = images du train officiel présentes dans tous les pipelines, utilisées en entraînement par aucun, et impaires (`--val-rule odd`).
- Familles comparées : chaque pipeline seul, moyenne pondérée des probabilités, moyenne en logit, un pipeline par régime (partiel / complet, `modal_t1/cache/global_*.json`, seuil 0,80), ou n'importe quel candidat par régime.
- Post-traitement repris du pipeline Modal (égalité testée avec `tracks/postproc/pp.py`) : seuil par régime, K plus grandes composantes et remplissage en option (`--pp thr,K,fill`), rectangle de `solution/cache/search_*` si score ≥ t, jamais de grille vide ni pleine.
- Tous les choix sont faits dans les plis d'entraînement d'une validation croisée répétée (10 × 5 plis) ; le tableau donne la moyenne, l'écart-type, un IC bootstrap à 95 % et le gain apparié sur le meilleur pipeline seul. Une famille d'assemblage n'est retenue que si son gain dépasse 0,005 avec une probabilité bootstrap ≥ 0,80 (`--min-gain`, `--min-p`).
- La métrique est `wmloc.utils.miou_image` du kit ; le test vérifie qu'elle est identique à la copie du kit dans `modal_t1/kit` et à `tracks/postproc/common.py::miou`.

### Mesuré avec les fichiers locaux (entrées de démonstration, pas un résultat d'assemblage)
- `make-demo` fabrique deux pipelines factices à partir du baseline et du baseline + rectangles. Validation propre : 23 à 27 images par outil (moitié impaire ∩ pli held-out de 150). CV : PixelSeal 0,521, TrustMark 0,608, StegaStamp 0,469.
- `submission_rect.pt` contient 222 grilles vides et 172 grilles pleines sur 1020 : ces images plafonnent à 0,5.

### Diagnostic du pipeline Modal (`modal_t1/diag/errors/per_image.json`, 271 images impaires)
Points perdus par outil (en points de mIoU de l'outil), par régime et forme :

| Outil (mIoU) | complet, trous en forme d'objet | partiel, objet | partiel, rectangle | partiel, barre |
|---|---|---|---|---|
| PixelSeal (0,572) | 0,57 → 19,5 pts perdus (n=39) | 0,59 → 11,9 (n=25) | 0,62 → 4,9 (n=11) | 0,51 → 4,6 (n=8) |
| TrustMark (0,707) | 0,61 → 15,9 (n=37) | 0,69 → 7,6 (n=22) | 0,88 → 3,5 (n=26) | aucune |
| StegaStamp (0,853) | 0,87 → 5,9 (n=43) | 0,85 → 3,2 (n=20) | 0,82 → 2,8 (n=15) | 0,87 → 1,4 (n=10) |

- Le régime est presque toujours bien décidé (3 erreurs sur 271) ; l'oracle de régime ne rapporte rien.
- Le meilleur seuil par image ne rapporte que +2 pts (PixelSeal, StegaStamp) : la perte vient de la carte, pas du seuil.
- Images complètes : plus le trou est petit, plus le score baisse (PixelSeal 0,65 pour 10-30 % de trou, 0,48 pour < 2 % ; TrustMark 0,66 → 0,51).
- Rectangles : TrustMark en utilise 24 sur 26 (0,71 → 0,88) ; PixelSeal 5 seulement, et l'oracle « rectangle ou réseau » ne vaut que +0,4 pt.
