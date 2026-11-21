# REPRISE — état au sam. 21 nov. 2026, 19:15

À lire en premier par toute nouvelle session. Détails par tâche : `challenge/task{1,2,3}_*_report.md`.
Fin du hacking : **dim. 22 nov. 12:00**.

## Leaderboard public
| Tâche | Soumis | Score public | Remarque |
|---|---|---|---|
| T1 watermark | (score actuel 0,26) | 0,26 | `submission_baseline.pt` fait 0,466 en local, **pas encore soumis : décision utilisateur en attente** |
| T2 model equality | rien | — | collecte des réponses `ref-*` en cours |
| T3 unlearning | `sub_r2_g1` (oubli fort) | **0,2384, 1er** | `sub_r2_g0` (oubli doux) = 0,0798 → l'hypothèse « oubli fort » est validée |
| T3 | `sub_r2_g2` (9 classes oubliées) | en attente | upload de 328 Mo expiré, relancé par l'ancienne session vers 19:11 |

Soumission : `repos/videoseal/.venv/bin/python scripts/submit.py <task> <fichier>` (clé lue dans `.env`, non versionné). Délai d'environ 10 min entre deux soumissions.

## Fichiers lourds (ignorés par git, présents sur ce Mac)
- `tasks/unlearning/candidates/sub_r2_g{0,1,2}.pt` : copies des candidats soumis (l'original était dans un dossier temporaire de session).
- `tasks/unlearning/candidates/local_mlp_*.pt` : candidats de secours niveau MLP.
- `tasks/Hackathon_Watermark_Localisation_Task/solution/submission_baseline.pt` : T1, 0,466 en local.
- Sur JURECA : `/p/scratch/training2666/geran1/IncomingPHd/unlearning/output/sub_*.pt` et `output/runs/<run>/best.pt`.

## JURECA (`ssh juelich`, connexion multiplexée déjà ouverte)
- File partagée : `/p/scratch/training2666/geran1/IncomingPHd/queue/{pending,running,done,failed,logs}/`.
- Workers `ipwork1-3` (jobs 15902413-15902415) : PENDING à 19:13, démarrage estimé vers 21:10.
- Tâches en attente : `10_t1_rect_train`, `11_t1_rect_test`, `30_t1_rect_fine_train` (T1, recherche de rectangle) ; `10_t3_r6`, `11_t3_r7`, `11z_t3_xstrong`, `12_t3_r5`, `13_t3_r3` (T3, sweeps autour de l'oubli fort).
- Suivi : `squeue -u $USER` ; résultats T3 : `cd .../unlearning && source .venv/bin/activate && python solution/collect.py`.

## Prochaines actions
1. **T3** : lire le score de `sub_r2_g2`. Quand les workers tournent, classer r6/r7/xstrong/r5/r3 avec `collect.py` (hypothèse `strong`, puis `xstrong`) et soumettre le meilleur `output/sub_<run>.pt`.
2. **T1** : demander à l'utilisateur s'il faut soumettre `submission_baseline.pt` (0,466 local contre 0,26 public). Puis rapatrier `solution/cache/search_*.npz` et lancer `rect_eval.py --test solution/submission_rect.pt` (voir rapport T1).
3. **T2** : `collect.py` remplit `tasks/model_equality/solution/data/ref_answers.jsonl` (181/450 à 19:12). Calibrer sur les `ref`, puis lancer la collecte `ep-*` (~6,5 h, à démarrer au plus tôt). Un seul process de collecte à la fois (clé partagée, 4 requêtes simultanées max).

## Attention au changement de compte
- L'ancienne session (ancien compte) a encore des processus actifs : `submit.py` pour `sub_r2_g2` et `collect.py` pour T2. Les laisser finir. Ne pas lancer un second `collect.py` en parallèle.
- Si l'ancienne session reprend toute seule à 19:40, ne garder **qu'une seule** session aux commandes.
