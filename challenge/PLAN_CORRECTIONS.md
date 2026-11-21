# Corrections au « Plan de bataille » (artifact partagé) — état réel au sam. 21/11, 21:25
Ce fichier fait foi en cas de conflit avec le plan. Il complète `challenge/STATUS.md` (scores) et `challenge/HANDOFF_COMPUTE.md` (reprise sur Modal ou autre machine).

## 1. Ce qui est DÉJÀ FAIT (ne pas refaire)
| Point du plan | État réel |
|---|---|
| Un score sur les 3 tâches | ✅ fait depuis 17:45 |
| T1 « version simple ce soir » (attendu ≈ 0,45) | ✅ **0,4584 public** : `tasks/Hackathon_Watermark_Localisation_Task/solution/submission_baseline.pt` (held-out 0,466) |
| T3 « garder la soumission » | ✅ **0,2736 public, 1re place** : `sub_m1_g2.pt` (mixcls, α=5) |
| T1 chasse au rectangle | 🟡 code prêt (`solution/rectsearch.py`, `run_search.py`, `rect_eval.py`, `job_rect*.sh`) ; caches `solution/cache/search_*.npz` versionnés ; voir §3 |
| T2 entraînement sur ref-01..09 | 🟡 273 réponses déjà collectées (`tasks/model_equality/solution/data/ref_answers.jsonl`, uniquement sur le Mac de geran1) ; pipeline complet dans `tasks/model_equality/solution/` |

## 2. Corrections factuelles
1. **La clé API n'est pas dans `~/Downloads/hackathon-watermark/.env`** : elle est dans **`.env` à la racine de ce dépôt** (`HACKATHON_API_KEY=...`, gitignoré).
   - Sur une autre machine, recréer ce fichier ; demander la clé au chef d'équipe en message privé.
   - Soumettre avec `python scripts/submit.py <task_id> <fichier>`.
2. **Budget T2 : aucune question dépensée.** Vérifié sur le serveur : 0/100 sur les 33 endpoints. On peut le revérifier gratuitement avec `GET http://34.147.229.55/33-model-equality/v1/models` (header `X-API-Key`).
3. **JURECA n'est PAS « en plus de Modal ».**
   - Règle des organisateurs (menace de révocation) : **1 seul job Slurm pour toute l'équipe** (en cours + en attente, tous membres confondus).
   - Notre unique job est le worker `ipwork3` (fin vers 23:15), qui exécute `queue/pending/*.sh` l'un après l'autre.
   - ⇒ **Tout le gros calcul de la nuit (200k exemples, U-Net, banc d'essai T2) va sur Modal.** Ne jamais lancer de `sbatch` ; sur JURECA, seulement déposer des scripts dans la file.
4. **Cooldowns de soumission** : T2 = 30 min ; T3 ≈ 10 min (constaté : « wait 455 s ») ; T1 non mesuré, sans doute ~10 min.
5. **Upload T3** : 328 Mo par fichier, et un envoi a déjà expiré une fois. Relancer en cas de timeout.

## 3. Failles à corriger en priorité
### T1 — 434 grilles VIDES sur 1020 dans la soumission actuelle ⚠️ (gain rapide le plus important)
- Aucune image n'a de zone vide, donc une grille vide = IoU₁ = 0 ⇒ mIoU de l'image ≤ 0,5.
- Toute méthode qui met au moins quelques bonnes cases sur ces 434 images (petites zones) gagne directement.
- C'est exactement la cible de la chasse au rectangle et du U-Net.
- **Règle à appliquer partout : jamais de grille vide.**
### T1 — état de la chasse au rectangle sur JURECA
- Incident disque « Disk quota exceeded » entre 20:46 et 20:49 (passager ; l'écriture refonctionne).
  - Échecs : `trustmark test 0/1` et `pixelseal test 1,2,4,5 /6`.
  - `queue/failed/` contient `02_t1_rect_all` et `03_t1_rect_all2`.
  - La tâche de correction `04_t1_rect_fix` est **terminée**.
- **Avant `rect_eval.py`** : vérifier que chaque `cache/search_test_*` contient bien toutes les images de test (340 par méthode).
- Logs : `Hackathon_Watermark_Localisation_Task/output/rect*.log` sur JURECA.
### T1 — U-Net
- Un premier U-Net (entraîné sur des données qui ne collaient pas exactement à celles des organisateurs) plafonnait à 0,455 held-out, sans gain par rapport à la décision de mode.
- Le plan a raison : **refaire les données avec le kit officiel** (`wmloc/watermarks.py` : trustmark==0.8.0 sans ECC, PixelSeal par défaut, StegaStamp 400 px + résidu redimensionné), zones partielles en 512×512, pas de JPEG, pas de flip ni de rotation.
### T3 — fp16
- Le serveur évalue en **demi-précision**, ce que nous n'avions pas vérifié.
- Vérification en cours : tâche `05_t3_fp16check` dans la file JURECA ; comparaison fp16/fp32 de `sub_m1_g2`, `sub_r2_g2`, `sub_r2_g1` sur la loss forget, la KL vs gold et l'accord retain.
### T3 — seuil 70 %
- Pousser l'oubli (α bas) aide au public, mais rapproche du seuil éliminatoire, que nous ne pouvons pas mesurer.
- Ne pas descendre α en dessous de ce que la vérification retain supporte.

## 4. Ce que le plan a bien vu (à garder)
- **T1** : 200k exemples générés avec le kit officiel (8× L4) + un U-Net par outil sur H100, puis un vote ; seuil réglé image par image.
- **T2** :
  - choisir les questions sur les ref ;
  - banc d'essai Modal avec plusieurs modèles ouverts ;
  - réponses longues ;
  - MMD famille → modèle ;
  - dépenser les 100 questions seulement après validation par l'utilisateur.
- **T3** :
  - COCO comme données « à garder » (plus proche que notre Imagenette actuel) : à tester sans toucher à la soumission ;
  - viser « jamais vue », pas zéro.
- **Gel à 11:00** : vérifier qu'il y a une soumission valide sur chacune des 3 tâches.

## 5. État des travaux en cours (21:25)
- **JURECA** : 1 job (`ipwork3`, fin vers 23:15). En cours : `11a_t3_alpha` (sweep α ∈ {2,3,4} + réplique). Ensuite : `05_t3_fp16check`, `11b_t3_alpha2`, `11bz_t3_soup` (model soup), `11c_t3_x3`, `11d_t3_x4`.
- **T1 et T2** : pilotées par la session Fable (terminal séparé de geran1).
- **T3** : pilotée par la session Opus (agent unlearning).
- Meilleurs candidats exportés sur JURECA : `/p/scratch/training2666/geran1/IncomingPHd/unlearning/output/sub_*.pt`.
