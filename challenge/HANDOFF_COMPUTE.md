# Reprise sur Modal / GPU externes — tout ce qu'il faut (sam. 21/11, 21:15)
À lire avec `challenge/STATUS.md` (scores et décisions), `challenge/00_tasks.md` (énoncé + règles), puis le rapport de la tâche visée (`challenge/task{1,2,3}_*_report.md`).

## Règles qui ne changent pas
- **Clé API** : dans `.env` (`HACKATHON_API_KEY=...`), jamais commitée et jamais affichée. Soumettre avec `python scripts/submit.py <task_id> <fichier>`.
  - Task ids : `34-watermark-localisation-image` (.pt), `33-model-equality` (.csv, 1 toutes les 30 min), `31-unlearning` (.pt state_dict, cooldown ~10 min).
- **Toujours demander l'accord de l'utilisateur avant une soumission**, et avant de dépenser le budget des endpoints ep-XX (Task 2, 100 réponses par endpoint pour toute la compétition, définitif).
- **JURECA** : 1 seul job Slurm pour toute l'équipe (règle organisateurs, sinon révocation). Sur Modal ou vos GPU, pas de limite.
- Fin des soumissions : **dimanche 22/11 à 12:00**.

## Données (publiques, sur Hugging Face) — à télécharger sur chaque machine
```bash
pip install -U huggingface_hub
hf download SprintML/Hackathon_Watermark_Localisation_Task --repo-type dataset --local-dir tasks/Hackathon_Watermark_Localisation_Task
hf download SprintML/model_equality --repo-type dataset --local-dir tasks/model_equality
hf download SprintML/unlearning --repo-type dataset --local-dir tasks/unlearning
```

## Task 1 — localisation de watermark (public 0,4584 ; leaders ≈ 0,55) — pilotée par la session Fable
- **Setup** (dossier de la tâche) :
  1. `pip install -r requirements.txt` (Python 3.11, torch 2.5.1 ; sans GPU NVIDIA, remplacer `onnxruntime-gpu` par `onnxruntime`) ;
  2. `unzip -q starter_kit.zip -d .` ;
  3. `python -c "from wmloc.data import unzip; unzip('train.zip','release'); unzip('test.zip','release')"` ;
  4. `python download_models.py` (internet requis ; TrustMark télécharge aussi ses poids au premier appel).
- **Soumis (0,4584)** : `solution/submission_baseline.pt`. Méthode : décision de mode globale (presque tout marqué ou presque propre) + seuillage des `per_position_logits`. Held-out 0,466.
- **Piste en cours, la recherche de rectangles** : environ la moitié des zones partielles sont des rectangles exacts (crop → embed → recollage).
  - Code : `solution/rectsearch.py`, `run_search.py`, `rect_eval.py`, `job_rect*.sh`.
  - Résultats en cache, versionnés : `solution/cache/search_*.npz` et `slide_*.npz`.
  - ⚠️ Les morceaux test `trustmark 0/1` et `pixelseal 1,2,4,5 /6` ont échoué pendant l'incident disque de JURECA (20:46-20:49) : **à vérifier ou relancer**.
- **Évaluation + soumission** : `python solution/rect_eval.py --test solution/submission_rect.pt` (affiche le held-out par méthode, puis écrit le fichier).

## Task 2 — model equality (public 0 ; leaders 0,40) — pilotée par la session Fable
- Pas de GPU nécessaire : uniquement de l'API et des statistiques. Code : `tasks/model_equality/solution/` ; plan complet et commandes : `challenge/task2_model_equality_report.md`.
- `ref-01..09` : gratuits, illimités, non scorés. **Calibrer dessus d'abord.** Il y a déjà 273 réponses dans `solution/data/ref_answers.jsonl` (non versionné, local au Mac).
- `GET /33-model-equality/v1/models` (gratuit) donne le budget restant par endpoint ; il était intact, 0/100 utilisé.

## Task 3 — unlearning ViT-B/16 (public 0,2736, 1re place)
- **Setup** : `pip install -r requirements.txt` (torch, timm) dans `tasks/unlearning/`.
  - Les proxies gold (`gold_*.pt`) et `orig_forget_logits.pt` sont versionnés dans `solution/cache/`.
  - Le retain proxy Imagenette (~1 Go) se reconstruit avec `solution/prep_retain.py` (internet requis).
  - Les gros caches `tokens_k*.pt` (5 Go chacun) ne servent qu'à la variante « tête seule » (`feat_unlearn.py`), pas à la méthode retenue.
- **Meilleur candidat** : `sub_m1_g2.pt`, public 0,2736. Il se trouve sur JURECA (`/p/scratch/training2666/geran1/IncomingPHd/unlearning/output/`) et en copie sur le Mac de geran1 ; il pèse 328 Mo, donc il n'est pas dans git. Il se recrée avec :
  ```bash
  cd tasks/unlearning/solution
  python -u unlearn.py --device cuda --amp --method td --blocks 12 --bs-f 64 --bs-r 64 --eval-every 100 \
    --n-eval 3500 --n-eval-r 1000 --target mixcls --select mixcls --alpha 5 --lr 5e-5 --steps 1500 --w-stat 0.1 \
    --out ../output/runs/m1_g2
  python make_submission.py ../output/runs/m1_g2/best.pt ../output/sub_m1_g2.pt
  ```
  Environ 10-15 min sur un A100 ; 4 configs en parallèle avec `solution/jureca_grid.sh <tag> "<cfg0>" ... "<cfg3>"` (lancer en `bash`, sans `sbatch`).
- **Ce que disent les scores publics** :

  | Cible | α | Score public |
  |---|---|---|
  | mixcls | 5 | **0,2736** |
  | mixcls | 10 | 0,2616 |
  | strong | — | 0,2384 |
  | cls | 20 | 0,2080 |
  | cls | 10 | 0,2046 |
  | medium | — | 0,0798 |

  ⇒ **cible mixcls + α bas** (= oubli plus fort).
- **Suite prévue** (déjà en file sur JURECA ; scripts dans `infra/jureca_queue/pending/`) :
  - `11a_t3_alpha` : α ∈ {2, 3, 4} + réplique d'α=5 avec une autre seed (mesure du bruit) ;
  - `11b_t3_alpha2` : d'autres seeds ;
  - `11bz_t3_soup` : model soup uniforme + greedy (`solution/soup.py`), pour réduire la variance sur les 70 % cachés.
  - Pour les relancer sur Modal, il suffit d'exécuter ces scripts (adapter le `cd`).
- **Risques** :
  - seuil des 70 % d'accuracy held-out : ne pas pousser α trop bas sans vérifier ;
  - l'évaluateur local est mal calibré par rapport au public (proxy 0,45-0,50 contre 0,27 public) : `solution/fit_public.py` le recale sur les scores publics.

## Fichiers JURECA utiles (accès : `ssh juelich`, dossier `/p/scratch/training2666/geran1/IncomingPHd/`)
- `unlearning/output/sub_*.pt` : tous les candidats Task 3 exportés ; `unlearning/output/runs/*/hist.json` : courbes.
- `Hackathon_Watermark_Localisation_Task/solution/cache/` : recherches de rectangles. `queue/` : file du worker (copie dans `infra/jureca_queue/`).
