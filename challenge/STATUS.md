# ÉTAT DE REPRISE (sam. 21/11 ~19:30) — à lire en premier
## Leaderboard (public)
- Task1 watermark: 0.2553 (all-zeros). PRÊT À SOUMETTRE (attend OK user): tasks/Hackathon_Watermark_Localisation_Task/solution/submission_baseline.pt (held-out 0.466, format vérifié).
- Task2 model equality: 0.0 (singletons). Pipeline: tasks/model_equality/solution (rapport challenge/task2_*). Étape suivante: calibration gratuite sur ref-01..09 (agent relancé, non confirmé), puis collecte ep (DÉPENSE TOUT LE BUDGET, irréversible) => demander OK user.
- Task3 unlearning: 0.2384 (sub_r2_g1, cible strong) = 1er. sub_r2_g0 (medium)=0.0798. sub_r2_g2 (mixcls) 'score': 0.2616.
## Clé API: .env (HACKATHON_API_KEY), jamais commitée. Soumettre: python scripts/submit.py <task_id> <file> (31-unlearning, 33-model-equality, 34-watermark-localisation-image). Cooldown ~10 min (T3), 30 min (T2).
## JURECA: ssh juelich (ControlMaster 12h; si fermé, user relance ssh juelich + MFA). Dossier: /p/scratch/training2666/geran1/IncomingPHd/
- 3 workers persistants ipwork1-3 (16h50, PENDING, démarrage estimé ~21:10) lisent queue/pending/*.sh (ordre lexical), logs queue/logs/, fin -> queue/done|failed. Script: queue/worker.sh.
- File: 10_t1_rect_train, 11_t1_rect_test, 30_t1_rect_fine_train (T1 recherche rectangles, cf. challenge/task1_*); 10_t3_r6, 11_t3_r7, 12_t3_r5, 13_t3_r3 (T3 sweeps strong; exports unlearning/output/sub_rX_gY.pt + collect.py dans logs; classer colonne strong= puis plus faible retainKL).
- T1 après tâches 10/11: rsync cache/search_*.npz puis rect_eval.py --test solution/submission_rect.pt (voir rapport T1).
## User: pas de limite de nœuds; demander OK avant chaque soumission et avant de dépenser le budget ep (T2). Coéquipiers: boulay1, chaibamensou1, kourdoughli1 (teammate.sh, TEAM_LEAD=geran1, TEAM_FOLDER=IncomingPHd).
