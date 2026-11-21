# Coordination Task 1 entre sessions (boîte aux lettres)

Chaque session ajoute ses messages en bas, datés, puis `git pull --rebase && git push` sur `develop`.
Gros fichiers : jamais dans git, toujours sur JURECA dans
`/p/scratch/training2666/geran1/IncomingPHd/Hackathon_Watermark_Localisation_Task/output/probs/`.

---

## 22/11 00:55 — session de Matis (JURECA) → session d'Amil (Modal)

### Où nous en sommes
- Ton pipeline est fusionné dans `develop`. Rapport lu : `challenge/task1_modal_report.md`.
- Notre pipeline : tête de segmentation sur l'extracteur officiel de chaque outil (`solution/train_seg2.py`), entraînement en cours sur JURECA, fin vers 01:15.
- Held-out après ~20 min (provisoire, pas encore sur ta moitié impaire) : PixelSeal ≈ 0,655, TrustMark ≈ 0,65, StegaStamp ≈ 0,64.
- Complémentarité : **ton StegaStamp (0,853) ≫ le nôtre ; notre PixelSeal (≈ 0,655) > le tien (0,571) ; TrustMark proches, à moyenner.**
- Nous avons adopté ta validation : images officielles impaires (`int(nom, 16) % 2 == 1`), jamais utilisées en entraînement.

### Ce qu'il nous faut de toi
1. Tes probabilités, au format d'échange. Depuis ton dossier `modal_t1`, après `git pull` :
   ```bash
   python <repo>/tasks/Hackathon_Watermark_Localisation_Task/solution/ensemble.py import-amil \
     --run pixelseal=pixelseal_research/runs/hires_c+pixelseal_research/runs/frozen_a,trustmark=runs/trustmark_mix,stegastamp=runs/stegastamp_mix \
     --out probs_export
   ```
   Cela écrit 6 fichiers `amil_<outil>_{train,test}.npz` (~10 Mo chacun). Dépose-les dans le dossier JURECA ci-dessus (compte `kourdoughli1`).
   La commande n'a été testée que sur des fichiers de démonstration : si elle échoue sur tes dossiers, colle l'erreur ici.
2. Si possible : `submission_t1_v1.pt`, tes masques SAM du test (`tracks/sam_shapes/sam/test_*.npz`), tes caches de rectangles (`cache_dl/search_*_f*.npz`).
3. Une confirmation : les images impaires n'ont servi à aucun de ces runs. Ne mélange pas ta deuxième série (fabriquée sur les impaires) sans le signaler ici, sinon nous perdons la validation commune.

### Ce que nous te donnons
- Nos probabilités seront au même format, même dossier JURECA : `seg2_<outil>_{train,test}.npz`, vers 01:20.
- Banc d'assemblage : `solution/ensemble.py` (validation croisée, post-traitement identique à ton `pp.py`, jamais de grille vide ni pleine, écriture de la soumission). Commande dans `challenge/task1_watermark_report.md`.
- Diagnostic de ton pipeline (depuis ton `per_image.json`) : la première perte est sur les images complètes à petit trou propre (PixelSeal et TrustMark), puis PixelSeal sur les zones fines.

### Règles communes
- Soumissions Task 1 : celui qui a un fichier meilleur en validation commune l'écrit ici avec son score, pour éviter les doublons.
- Task 2 : nous ne lançons aucune collecte `ep`, la tienne tourne (lu dans ton rapport).
- JURECA : un seul job Slurm pour toute l'équipe, c'est le nôtre (`ipwork4`). Ne fais pas de `sbatch`.

---
