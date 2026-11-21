# European AI Championship Paris — hackathon 24h (21-22 nov. 2026, Epitech KB)

- Hacking : sam. 14:30 → dim. 12:00 (pitch 6 min pour le top 6 du leaderboard).
- **ÉNONCÉ OFFICIEL : `challenge/00_tasks.md` (3 tâches : localisation de watermark, model equality testing, unlearning ViT) + `challenge/01_organizer_audio.md` (consignes orales). À lire avant tout travail.**
- **ÉTAT ET REPRISE (scores, ce qui tourne, corrections du plan, Modal) : `challenge/STATUS.md`, `challenge/PLAN_CORRECTIONS.md`, `challenge/HANDOFF_COMPUTE.md`.**
- Équipes de 3-4. Organisé par CISPA + PoC Innovation, partenaires Meta/Epitech.
- Papiers de prépa : voir `papers/README.md`. Avant de travailler sur un challenge lié au watermarking, à l'audit d'API LLM ou au machine unlearning, lire le fichier correspondant dans `papers/context/` :
  - watermarking image/vidéo → `papers/context/06_organizer_talk_watermarking.md` (talk de l'organisateur : priorités du jury) puis `01_pixel_seal.md` ; fiches de la lignée : `05_hidden`, `07_watermark_anything`, `08_video_seal`, `09_synthid_image`, `10_trustmark`, `11_editguard`, `12_stable_signature`, `13_tree_ring`, `14_distseal` (index dans `papers/README.md`)
  - audit / identification de modèle derrière une API, tests statistiques → `papers/context/02_model_equality_testing.md`
  - machine unlearning, membership inference attacks → `papers/context/03_scrub_unlearning.md`
- Code officiel des 3 papiers installé dans `repos/` (un venv uv par repo, smoke tests dans `repos/_smoke/`) : lire `repos/README.md` avant de l'utiliser (pièges d'installation et API).
- **RÈGLE ORGANISATEURS JURECA (sinon révocation de l'accès GPU) : toute l'équipe = AU PLUS 1 job Slurm à la fois (en cours + en attente), 1 nœud = 4 GPU. Vérifier `squeue -h -u geran1,boulay1,chaibamensou1,kourdoughli1` avant tout sbatch. Ne pas demander des durées de 16-18 h. On utilise UN worker (`queue/worker.sh`) qui exécute `queue/pending/*.sh` l'un après l'autre ; ne jamais lancer de 2e job.**
- Compute GPU (cluster JURECA, A100 40 GB, Slurm, soumission leaderboard `id,label` en probabilité [0,1]) : lire `infra/jureca_gpu.md` avant de lancer un entraînement ou une soumission.
- **PRIORITÉ : les 3 papiers initiaux (01, 02, 03) sont le cœur du projet.** Le MVP doit s'appuyer d'abord sur eux (méthodes, code officiel, chiffres).
- BONUS seulement : `papers/context/04_complementary_papers.md` (attaques de watermark, fingerprinting LLM, évaluation de l'unlearning ; infos issues des abstracts). À utiliser uniquement pour compléter un point faible d'un des 3 papiers, jamais comme base du MVP, et ne pas le charger par défaut.
- Sous-agents : passer le chemin du fichier `papers/context/...` pertinent dans leur prompt. Détails supplémentaires : `papers/notes/` puis texte brut `papers/txt/` (grep).
