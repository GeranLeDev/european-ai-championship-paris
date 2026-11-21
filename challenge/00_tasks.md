# Les 3 tâches du challenge (slides officielles, 21/11/2026)
Source : `EuroAIChampionshipParis2026-11-21.pdf`, 78 slides, par Adam Dziedzic (CISPA) et Franziska Boenisch (TUM / Fraunhofer AISEC), équipe SprintML.
Le thème commun est **la traçabilité des données (data provenance) pour l'IA générative**, avec en toile de fond l'AI Act (art. 50 et 53).
Synthèse du vocal des organisateurs : `01_organizer_audio.md`.

## Planning
- **Sam. 21** :
  - 14:30 : début du hacking ;
  - 17:00 : dîner.
- **Dim. 22** :
  - 08:00 : petit-déjeuner ;
  - **12:00 : fin du hacking** ;
  - 13:00 : **présentations de 6 min, réservées aux 6 meilleures équipes du leaderboard** ;
  - 14:00 : délibération du jury ;
  - 14:30 : remise des prix.
- **1er prix : 10 000 €.**
- C'est une étape régionale (Paris) d'un championnat en 6 villes, avec une **grande finale au CISPA**.
- Communication : Discord https://discord.gg/urUG53tnbJ
- Classement par leaderboard. **Ne pas sur-optimiser le score public** : il ne couvre qu'une partie des données, et le classement final se fait sur un jeu de test caché.

## Classement général (page « How scoring works » du leaderboard)
- Pour chaque tâche, seule la **meilleure** soumission de l'équipe compte. Il y a un cooldown entre deux soumissions sur la même tâche ; pour la Task 2, c'est 30 min.
- **Normalisation par tâche** : `n_t = (s_t − min_t)/(max_t − min_t)`, calculée sur les meilleurs scores de toutes les équipes. La meilleure équipe obtient 1, la moins bonne 0 ; si toutes les équipes ont le même score, chacune a 0,5.
- **Score global = moyenne des n_t sur les 3 tâches.**
- **Seules les équipes qui ont un score sur LES 3 tâches sont classées** : sans soumission valide sur une tâche, on n'est pas classé.
- **Public = environ 30 %** des données d'évaluation, affiché pendant l'event. **Final = les 70 % restants**, jamais vus.
- **Conséquences** :
  1. **Soumettre tôt une baseline valide sur chacune des 3 tâches**, pour être classé.
  2. Les tâches pèsent **autant** l'une que l'autre, et la normalisation est relative aux autres équipes. Passer de dernier à médian sur une tâche faible rapporte autant que gagner des points sur une tâche forte : **ne délaisser aucune tâche**.
  3. Le public ne couvre que 30 % des données : choisir les hyperparamètres par validation locale, pas sur le score public.

---

## TASK 1 — Localiser le watermark dans une image
- **Question** : quelle partie de l'image porte un watermark ?
- **Méthodes de watermark** : **PixelSeal**, **TrustMark**, **StegaStamp**.
- **Format** :
  - images de 512×512 ;
  - on prédit sur une **grille de 128×128 cellules de 4×4 pixels** ;
  - label 1 si au moins 50 % de la cellule est marquée, 0 sinon.
- **Métrique** :
  - IoU par classe (marquée / propre) ;
  - **mIoU = (IoU₀ + IoU₁)/2** par image ;
  - score final = moyenne sur les images.
- **Données** : 1530 images, réparties à parts égales entre les 3 méthodes.
  - **Train : 510 images** (170 par méthode), avec le masque vrai (512² et 128²), la méthode et le **message embarqué (bits)**.
  - **Test : 1020 images** (340 par méthode), avec l'image, **la méthode et le message embarqué**.
- **Pistes** :
  - Le message est connu au test. On peut donc **re-marquer l'image avec le même message** (pour PixelSeal et TrustMark, dont les poids sont dans `repos/videoseal/ckpts/`), puis comparer à l'image donnée : un résidu cohérent indique une zone marquée.
  - Autre option : un **décodage local** (fenêtres glissantes) avec mesure de la bit accuracy par patch par rapport au message connu.
  - Le modèle **WAM** fait de la segmentation pixel par pixel (fiche `papers/context/07_watermark_anything.md`). On peut le fine-tuner sur les 510 masques comme prédicteur de masque, conditionné par la méthode.
  - Un **petit U-Net / segmenteur** entraîné sur le résidu (image − version débruitée), avec un modèle par méthode.
  - Fiches à lire : `06`, `01`, `07`, `10`. StegaStamp n'est **pas** dans nos baselines : il faut le récupérer (repo de Tancik et al.).

## TASK 2 — Model Equality Testing : deux APIs servent-elles le même modèle ?
- **33 endpoints** (ep-01 … ep-33), soit 528 paires.
- **Variantes de déploiement** :
  1. modèle de base ;
  2. config de quantization ;
  3. adaptateur LoRA ;
  4. base + watermark ;
  5. base + system prompt ;
  6. base + stratégie de décodage (greedy ou beam).
  - Les variantes 1 à 3 sont des **identités de modèle différentes**.
  - Les variantes 4 à 6 comptent comme le **MÊME modèle** que la base.
- **But** : pour chaque endpoint, trouver sa famille (même modèle de base), puis le modèle exact.
- **Score** :
  - **0,1 × MCC_famille + 0,9 × MCC_modèle**, où MCC_x = max(0, MCC) ;
  - MCC_modèle porte sur les paires d'une même famille ;
  - deviner seulement la famille rapporte au plus 0,1 : **tout l'enjeu est de séparer les variantes au sein d'une famille**.
- **Budget** :
  - **100 réponses par endpoint pour toute la compétition** ;
  - ≤ 256 tokens générés et ≤ 1024 tokens d'entrée par prompt ;
  - 4 requêtes simultanées au maximum ;
  - `temperature` et `seed` sont ignorés, les system messages sont rejetés ;
  - les requêtes invalides ne sont pas décomptées.
- **Soumission** :
  - CSV `endpoint,family,model` (ex. `ep-01,A,A1`) ;
  - **1 soumission toutes les 30 min**.
- **Pistes** (fiche `papers/context/02_model_equality_testing.md`) :
  - **Test MMD à noyau Hamming** sur des complétions (code dans `repos/model-equality-testing/`), à appliquer paire par paire au sein d'une famille.
  - **Le budget est minuscule** (100 réponses par endpoint) : il faut choisir les prompts avec soin et **réutiliser les mêmes prompts pour tous les endpoints**, afin de comparer les distributions à prompt fixé.
  - **Pièges : les variantes « même modèle »** (watermark, system prompt, décodage greedy ou beam) changent la distribution de sortie sans changer l'identité du modèle.
    - Un test « distributions égales » les déclarerait différentes à tort.
    - Il faut des signatures qui sont **invariantes au décodage et au system prompt**, mais **sensibles aux poids** (quantization, LoRA) :
      - réponses factuelles ou déterministes ;
      - complétions à forte confiance ;
      - erreurs caractéristiques ;
      - longueur et tokenisation.
  - Clustering : regrouper d'abord en familles (style, tokenizer, connaissances), puis subdiviser.

## TASK 3 — Unlearning d'un classifieur d'images
- **Setup** :
  - un **ViT-B/16** pré-entraîné sur un dataset privé de 100 classes ;
  - **3500 images à oublier**, fournies **sans labels** ;
  - on soumet le `state_dict()` du modèle désappris.
- **Modèles de référence** :
  - `orig` : le modèle fourni ;
  - `gold` : la moyenne de plusieurs modèles **ré-entraînés sans le forget set**. C'est la cible.
- **Porte éliminatoire : si l'accuracy held-out est < 70 %, le score vaut 0.**
- **Score final = forget_score × retain_score.**
  - `forget_score = mean_std_score × prediction_score` :
    - `mean_std_score = 1 − clip(d_sub/d_orig, 0, 1)`, avec d = |Δ moyenne de loss| + |Δ écart-type de loss| par rapport à gold, sur le forget set ;
    - `prediction_score = 1 − clip(KL(gold‖sub)/KL(gold‖orig), 0, 1)`, calculé par image avec un softmax à **T = 2**.
  - `retain_score = retain_accuracy_score × retain_kl_similarity` :
    - accuracy : clip(acc_sub/acc_orig, 0, 1) sur un control set held-out ;
    - KL : 1/(1 + moyenne de KL(orig‖sub)).
- **Fichiers fournis** :
  - `model.pt` (poids seulement) ;
  - `forget_images.npz` (3500 images) ;
  - `task_template.py` : builder, preprocessing serveur, helper de prédiction et stub `unlearn()` ; il sauve un `submission.pt` de référence ;
  - `submission.py`.
- **Pistes** (fiche `papers/context/03_scrub_unlearning.md`) :
  - **SCRUB, mais calibré.** Le but n'est **pas** de faire chuter l'accuracy sur le forget set : il faut **imiter gold**, qui garde des sorties « comme sur des données jamais vues ».
    - Un oubli trop agressif (loss trop haute) est pénalisé par `mean_std_score` et par la KL.
    - Notre smoke test montrait justement ce sur-oubli (MIA à 0,985).
  - Les labels du forget set sont absents : on utilise **les prédictions d'`orig` comme pseudo-labels**.
  - **Le retain set n'est pas fourni** : il faut une distillation `orig → sub` sur des données proxy pour garder la KL faible. Pistes :
    - des images augmentées ;
    - un autre dataset (ImageNet/CIFAR-100) ;
    - le forget set lui-même avec du bruit.
  - **Cible réaliste pour gold** : une loss sur le forget set ≈ la loss d'`orig` sur des données jamais vues de même difficulté.
    - On peut l'estimer sur des images augmentées ou hors distribution.
    - Ensuite, on vise cette moyenne et cet écart-type, par exemple avec **une montée de gradient plus un arrêt précoce sur la statistique de loss**, ou un adoucissement par température.

---

## Infra GPU (mise à jour par rapport à `infra/jureca_gpu.md`)
- **Projet `training2666`** (confirmé). Chacun utilise **son propre compte Jülich**.
- **Setup d'équipe en 2 scripts**, à récupérer sur le **dataset Hugging Face `SprintML/hackathon`**. Instructions d'accès dans les descriptions des tâches, puis **suivre le README du repo à la lettre**.
  1. **Le team lead** lance `hackathon_setup.sh` **une seule fois**.
     - Il remplit d'abord `TEAM_LEAD`, `TEAMMATE_1..3` (vrais usernames Jülich) et `TEAM_FOLDER`.
     - Le script active le projet, crée le dossier partagé, télécharge les datasets, installe uv et crée un `.venv` par tâche, avec des caches uv et HF partagés.
  2. **Chaque coéquipier** lance `teammate.sh` **après** le lead, avec le même `TEAM_LEAD` et le même `TEAM_FOLDER`. Il ne retélécharge rien.
  - **Ne jamais lancer les deux scripts depuis le même compte.**
- **Dossier partagé** : `/p/scratch/training2666/<lead>/<team>/`, avec `unlearning/`, `model_equality/`, `Hackathon_Watermark_Localisation_Task/`, `.uv/` et `.cache/`. Chaque tâche a un `.venv`, un `requirements.txt` et un `output/`.
- **À chaque login** :
  ```bash
  jutil env activate -p training2666
  cd /p/scratch/training2666/<lead>/<team>/<task_folder> && source .venv/bin/activate
  ```
- **`main.sh`** :
  ```bash
  #SBATCH --account=training2666
  #SBATCH --gres=gpu:1
  #SBATCH --partition=dc-gpu
  #SBATCH --reservation=cispahack
  #SBATCH --output=output/%j.out
  srun python code.py
  ```
  Faire `mkdir -p output` d'abord, puis `sbatch main.sh` ; suivre avec `squeue -u $USER`.
