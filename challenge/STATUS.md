# ÉTAT DE REPRISE (sam. 21/11 ~19:30) — à lire en premier
**Lire aussi : challenge/PLAN_CORRECTIONS.md (corrections du plan partagé) et challenge/HANDOFF_COMPUTE.md (reprise sur Modal/autre machine).**

## PRIORITÉS (sam. 22:45, leaderboard public)
- Nous: T1 0.46 (leaders 0.75, ~10 équipes 0.69-0.75) | T2 0.00 (leaders 1.00, plusieurs 0.70-0.85) | T3 0.29 (leader 0.40). Overall estimé ~0.35-0.40, hors top 9.
- 1) T2 = plus gros gain (0 -> ~0.7 ≈ +0.2 overall). Finir calibration ref et DÉPENSER le budget ep ce soir (avec OK user). Le 1.00 d'autres équipes prouve un signal net (logprobs ? prompt_tokens ? déterminisme ?).
- 2) T1: 0.46 -> ~0.70 atteignable. 434 grilles vides dans notre soumission. Segmenteur entraîné sur données kit officiel (Modal) + chasse rectangles.
- 3) T3: plus faible levier. Plus de pas = gain (st_* 4000 pas proxy 0.57 vs 0.50). En cours de soumission: sub_st_g3, sub_st_g1.
- NŒUD JURECA: réservé à la Task 1 après 09_t3_long (plus aucune tâche t3 à mettre en file).

## Leaderboard (public)
- Task1 watermark: 0.4584 public (submission_baseline.pt, soumis 19:20). Suite: recherche rectangles (queue t1) -> submission_rect.pt.
- Task2 model equality: 0.0 (singletons). Pipeline: tasks/model_equality/solution (rapport challenge/task2_*). Étape suivante: calibration gratuite sur ref-01..09 (agent relancé, non confirmé), puis collecte ep (DÉPENSE TOUT LE BUDGET, irréversible) => demander OK user.
- Task3 unlearning: BEST 0.2893 (sub_ma_g0 mixcls α=2). ma_g1 α3 0.2806, m1_g2 α5 0.2736, r2_g2 α10 0.2616, r2_g1 strong 0.2384, m3_g3 cls 0.2080, m3_g2 cls 0.2046, r2_g0 medium 0.0798. => mixcls + α bas (monotone). fp16 vérifié OK.
## Clé API: .env (HACKATHON_API_KEY), jamais commitée. Soumettre: python scripts/submit.py <task_id> <file> (31-unlearning, 33-model-equality, 34-watermark-localisation-image). Cooldown ~10 min (T3), 30 min (T2).
## JURECA: ssh juelich (ControlMaster 12h; si fermé, user relance ssh juelich + MFA). Dossier: /p/scratch/training2666/geran1/IncomingPHd/
- RÈGLE: 1 seul job Slurm pour toute l équipe. 1 worker (ipwork3, 15902415, limite 4h depuis 19:17) lit queue/pending/*.sh (ordre lexical), logs queue/logs/, fin -> queue/done|failed. Quand il se termine, en relancer UN seul (sbatch --time=04:00:00 queue/worker.sh) après avoir vérifié qu aucun job équipe ne tourne.
- File: 10_t1_rect_train, 11_t1_rect_test, 30_t1_rect_fine_train (T1 recherche rectangles, cf. challenge/task1_*); 10_t3_r6, 11_t3_r7, 12_t3_r5, 13_t3_r3 (T3 sweeps strong; exports unlearning/output/sub_rX_gY.pt + collect.py dans logs; classer colonne strong= puis plus faible retainKL).
- T1 après tâches 10/11: rsync cache/search_*.npz puis rect_eval.py --test solution/submission_rect.pt (voir rapport T1).
## User: pas de limite de nœuds; demander OK avant chaque soumission et avant de dépenser le budget ep (T2). Coéquipiers: boulay1, chaibamensou1, kourdoughli1 (teammate.sh, TEAM_LEAD=geran1, TEAM_FOLDER=IncomingPHd).
