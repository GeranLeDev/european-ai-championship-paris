# Questions probables du jury, et nos réponses

Le jury (CISPA / SprintML, Meta) « grille » les 6 meilleures équipes. **La note finale combine les scores et la qualité des explications** : chaque membre doit pouvoir répondre sur chaque tâche.
Les questions sont en anglais, comme le jury les posera. Les réponses sont en français, à dire en anglais.
`[À COMPLÉTER]` = un chiffre ou un détail à récupérer avant le pitch.

_Mis à jour le dim. 22/11 à partir de `main` + `develop` (état ~02:35). Sources : `challenge/task1_modal_report.md`, `challenge/task1_watermark_report.md`, `challenge/COORD_T1.md`, `challenge/task2_model_equality_report.md`, `challenge/task3_unlearning_report.md` (§8 sur `develop`)._

---

## Nos chiffres, à connaître par cœur

| Tâche | Progression publique | Meilleur fichier | Validation locale |
|---|---|---|---|
| T1 watermark | 0,26 → 0,4584 → 0,7065 → 0,7523 → **0,7656** → v4c [À COMPLÉTER] | `modal_t1/submission_t1_v4c.pt` (soumis 01:28) | moitié impaire du train officiel : 0,784 pour v4c |
| T2 model equality | 0,00 → [À COMPLÉTER] | CSV issu de `fingerprint.py` | ref-01..09 : 0,472 ; simulateur : 0,760 |
| T3 unlearning | 0,0798 → 0,2384 → 0,2616 → 0,2736 → 0,2893 → **0,3045** | `sub_lg_g0.pt` (mixcls, α=3, 8000 pas) | proxy : retain accord 0,861 / KL 0,047 |

Calibration T1 local → public : 0,705 → 0,7065 ; 0,750 → 0,7523 ; 0,767 → 0,7656. **La validation locale prédit le public à ±0,002.**

---

## Questions générales (sur les 3 tâches)

**Q1. "Is this solution really yours? Walk us through this line."**
- Fichiers à savoir expliquer :
  - Task 1 : `modal_t1/train_lib.py` (U-Net), `modal_t1/pixelseal_research/det_head.py` (tête sur le détecteur PixelSeal), `solution/rectsearch.py` (recherche de rectangle), `modal_t1/combine2.py` (assemblage) ;
  - Task 2 : `fingerprint.py` (familles et modèles), `tokprobe.py` (empreinte du tokenizer), `collect.py` (collecte robuste) ;
  - Task 3 : `unlearn.py` (target distillation), `scoring.py` (score local), `fit_public.py` / `fit12.py` (hypothèses sur gold).
- Si on a utilisé un assistant de code, le dire honnêtement, mais montrer qu'on comprend chaque choix : *les idées viennent de nos expériences* (hypothèse → test → résultat). Nos rapports gardent aussi **ce qui a échoué** (quiz T2, U-Net PixelSeal, fit T3 hors famille).

**Q2. "How do you know you didn't overfit the public leaderboard?"**
- Task 1 :
  - validation = **moitié impaire** des images officielles du train (`int(nom, 16) % 2 == 1`), jamais utilisée pour fabriquer des exemples d'entraînement ;
  - le local prédit le public à ±0,002 sur trois soumissions (tableau ci-dessus) ;
  - on ne soumet que si le gain local dépasse 1 point ;
  - **réserve honnête** : la moitié impaire a servi à choisir le meilleur pas et le seuil, donc elle est un peu optimiste. Pour v4c, on a corrigé : deux séries de réseaux (A entraînée à partir des images paires, B à partir des impaires), chaque image de validation n'est prédite que par la série qui ne l'a pas vue.
- Task 2 : réglages choisis sur ref-01..09 et sur notre simulateur (27 endpoints). Pas de jeu de validation indépendant, on le dit. Seul contrôle hors échantillon : les 22 derniers prompts ref, collectés après le choix des seuils.
- Task 3 : c'est la question délicate, voir Q14. On a 14 points publics et on a **mesuré** que notre ajustement sur le leaderboard sur-apprenait (prédit 0,335, obtenu 0,283). On a alors arrêté de s'en servir pour choisir des candidats hors de la recette validée.

**Q3. "What would you do with more time / compute?"**
- Task 1 : les images **complètes avec de petits trous propres** coûtent 45 % de nos points perdus (PixelSeal et TrustMark). Il faut un générateur qui reproduit exactement ces trous (champs de taches lisses, masques rugueux) et le cadre décalé de TrustMark (`gen_lib_v2.py`, en cours).
- Task 2 : dépenser la seconde moitié du budget (passe 2) sur les familles ambiguës. Notre courbe score / budget ne plafonne pas avant 80 réponses par endpoint.
- Task 3 : un vrai retain **in-distribution** (DeepInversion sur `orig`, ou CutMix d'images forget), et une information par image sur gold, que les scores publics ne donnent pas.

**Q3b. "You used paid cloud compute (Modal): is your result reproducible on the provided cluster?"**
- Le calcul est le même : JURECA (A100 40 Go) suffit. Modal nous a fait gagner du temps parce que la règle des organisateurs limite chaque équipe à **un seul job Slurm** sur JURECA.
- Ce qui a tourné sur Modal : 2 × 25 000 exemples fabriqués par outil, un U-Net par outil (H100, 12 000 pas, lot de 24), jusqu'à ~10 GPU en parallèle.
- [À COMPLÉTER : temps de génération et d'entraînement, et équivalent A100]

**Q4. "What is the link with the AI Act?"**
- **Art. 50** : marquer et détecter les contenus générés → Task 1, savoir *où* une image a été modifiée ou marquée.
- **Art. 53** : transparence des fournisseurs de modèles → Task 2, vérifier qu'une API sert bien le modèle annoncé.
- **RGPD, droit à l'effacement** → Task 3, retirer l'influence de données d'un modèle sans le ré-entraîner.

---

## Task 1 : localisation de watermark

**Résumé de notre méthode, à savoir par cœur :**
1. **Régime global.** On décode le message sur l'image entière : moins de 80 % de bits relus = image « partielle » (petite zone marquée), sinon « complète » (presque tout marqué). Seulement **3 erreurs sur 271** images de validation.
2. **Rétro-ingénierie de la recette des organisateurs** (diagnostics `modal_t1/diag/`) :
   - rectangles et barres = morceau exact **découpé, marqué, recollé** ;
   - zones irrégulières des images partielles = **masques SAM de l'image elle-même** (IoU médian 0,91) : l'image entière est marquée, puis seuls les pixels du masque sont collés ;
   - trous des images complètes = champs de taches lisses (64 %) ou masques rugueux (36 %), jamais de rectangle.
3. **Données fabriquées avec les embedders du kit officiel**, mêmes versions que les organisateurs : 25 000 exemples sur COCO + 25 000 fabriqués **à partir des images officielles paires**. Le mélange des deux est nettement meilleur que chacun seul (TrustMark 0,60 → 0,66).
4. **Segmenteur par outil :**
   - TrustMark, StegaStamp : U-Net EfficientNet-B4/B6, entrée 6 canaux = image + micro-détails (image − flou × 20), **image agrandie x2** avant le réseau ;
   - PixelSeal : le U-Net n'apprend rien (0,27). Ce qui marche : une **petite tête de segmentation sur le détecteur officiel PixelSeal gelé** + l'accord par position avec le message connu (0,665).
5. **Recherche de rectangle** (`rectsearch.py`) : on décode exactement chaque boîte candidate et on garde celle qui maximise l'accord avec le message. Elle remplace la carte du réseau si son score dépasse un seuil par outil (TrustMark 0,60 ; PixelSeal 0,35 ; jamais pour StegaStamp).
6. **Moyenne de plusieurs réseaux**, et **jamais de grille vide ni pleine**.

**Résultats (moitié impaire, mIoU officiel) :**

| Version | PixelSeal | TrustMark | StegaStamp | Moyenne | Public |
|---|---|---|---|---|---|
| Baseline (régime + seuillage des logits) | 0,479 | 0,459 | 0,460 | 0,466* | 0,4584 |
| + recherche de rectangle | 0,528 | 0,573 | 0,460 | 0,520* | non soumis |
| v1 : U-Net ResNet-50 + rectangles | 0,571 | 0,692 | 0,852 | 0,705 | 0,7065 |
| v3a | | | | 0,767 | 0,7656 |
| v4c : B6/B4 + x2, deux séries croisées | 0,668 | 0,759 | 0,925 | **0,784** | [À COMPLÉTER] |

\* fold held-out de 150 images (ancienne validation).

**Q5. "Why does decoding the exact crop work, and isn't that exploiting a dataset artifact?"**
- C'est une **propriété du marquage**, pas une fuite :
  - PixelSeal et TrustMark redimensionnent l'image avant d'insérer le message (256 px pour PixelSeal, 245 px pour TrustMark) ;
  - un watermark inséré sur un crop n'est donc lisible qu'**à la même échelle et au même cadrage**.
- Vérifié (`croptest.py`, `basin.py`) : sur la boîte exacte, 90-100 % de bits corrects contre ~50 % ailleurs. Tolérance au décalage : ±8-16 px pour TrustMark, ±2-4 px pour PixelSeal et StegaStamp.
- C'est ce que ferait un vrai auditeur : le fournisseur connaît son message et son décodeur, et cherche *où* il est lisible.
- **Mais ce n'est plus notre méthode principale** : elle ne couvre que les rectangles (25 % des images). Le gros du score vient du segmenteur appris. TrustMark utilise le rectangle sur 24 rectangles sur 26 (0,71 → 0,88 sur ces images) ; pour PixelSeal, l'oracle « rectangle ou réseau » ne rapporte que +0,4 point.

**Q5b. "Why a synthetic dataset, and how do you make sure it matches the real one?"**
- 510 images officielles seulement, la moitié avec ~4 % de zone marquée en médiane : trop peu pour un segmenteur.
- On reproduit la recette **mesurée** (point 2 du résumé) avec **les embedders du kit officiel** (mêmes versions de `trustmark` et `torch`).
- Contrôle de fidélité : la trace TrustMark officielle est 6 fois moins corrélée à notre `embed` qu'en simulation. Un balayage du décodeur montre que les organisateurs ont marqué dans un **cadre 5-10 % plus grand** que l'image. On refabrique les exemples avec un cadre agrandi de 0 à 20 %.
- La validation se fait **uniquement sur les vraies images**, jamais sur du synthétique.

**Q5c. "Why no flip/rotation augmentation?"**
- Un watermark retourné ou tourné n'est **plus un watermark valide**. Le U-Net apprendrait des motifs qui n'existent pas au test.
- On n'augmente que l'image *avant* le marquage, jamais le résultat marqué.

**Q5d. "Why upsample x2 before the network? Why the high-pass channel?"**
- Le watermark est un signal de très faible amplitude et haute fréquence. Agrandir x2 donne plus de pixels de réseau par motif : c'est **le plus gros levier par réseau** (TrustMark B4 0,697 → 0,738 ; StegaStamp 0,893 → 0,926).
- Le canal micro-détails (image − flou, × 20) est indispensable : sans lui, le calcul en bf16 **efface la trace** par arrondi.

**Q6. "Is it realistic to assume the message is known?"**
- Oui pour le cas d'usage principal : **le fournisseur vérifie son propre watermark** (Meta, Google SynthID…). Il connaît sa clé et son message.
- L'énoncé fournit d'ailleurs le message et la méthode au test. On l'utilise pour le régime, la recherche de rectangle et la tête PixelSeal.

**Q7. "What about irregular (non-rectangular) masks?"**
- C'est le rôle du segmenteur appris. Points perdus par forme (pipeline v1, moitié impaire) :

| Outil | complet, trous | partiel, objet | partiel, rectangle | partiel, barre |
|---|---|---|---|---|
| PixelSeal | 0,57 (n=39) | 0,59 (n=25) | 0,62 (n=11) | 0,51 (n=8) |
| TrustMark | 0,61 (n=37) | 0,69 (n=22) | 0,88 (n=26) | — |
| StegaStamp | 0,87 (n=43) | 0,85 (n=20) | 0,82 (n=15) | 0,87 (n=10) |

- Point faible : les **images complètes à petit trou propre** (PixelSeal 0,48 pour moins de 2 % de trou).
- Le meilleur seuil par image ne rapporte que +2 points : la perte vient de la carte, pas du seuil.

**Q8. "Why is PixelSeal harder than TrustMark?"**
- PixelSeal module la force du watermark par une **carte JND** : quasi nul dans les zones plates (ciel, mur), fort sur les textures.
- Un U-Net seul n'y voit rien (0,27). D'où la tête sur le **détecteur officiel gelé**, qui sait déjà où lire le signal, plus l'accord avec le message connu (0,665).
- Les rectangles fins (un côté < 20-30 px) ne donnent **aucun accord**, même en décodant la boîte exacte : trop peu de pixels pour 256 bits.

**Q9. "How did you handle StegaStamp?"**
- C'est notre meilleur outil (0,925) : sa trace est forte et le U-Net B4 + x2 l'apprend bien.
- La recherche de rectangle, elle, échoue : le plancher de bruit du score est à 0,30-0,46 et un seul rectangle est retrouvé sur 23. On ne l'utilise pas.
- Détail technique : on a coupé le graphe ONNX fourni pour ne garder que le décodeur (4 fois plus rapide, sorties identiques).

**Q10. "Two teams worked on Task 1. How did you combine them?"**
- Deux pipelines : U-Net sur Modal, et sur JURECA une **tête de segmentation sur l'extracteur officiel affiné** (`train_seg2.py --arch feat` : ConvNeXt du détecteur PixelSeal, ResNet-50 du décodeur TrustMark ; 2 échelles 256 et 512 ; accord par position avec le message en canal ; générateur en ligne `wmgen.py` ; 70-75 min sur A100).
- Format d'échange commun (probabilités 128 × 128 par image) et banc d'assemblage `ensemble.py` : validation croisée 10 × 5 plis, une combinaison n'est retenue que si son gain dépasse 0,005 avec une probabilité bootstrap ≥ 0,80.
- Une même validation pour les deux équipes : la moitié impaire, jamais en entraînement.
- Mesuré (validation propre commune, 02:30) :
  - PixelSeal : seg2 **0,698** (n = 23), meilleur que la tête Modal (0,668). Moyenne en logit des deux : 0,707 (+0,009, IC [+0,002 ; +0,018]). C'est l'outil où l'assemblage paie, parce que les méthodes diffèrent ;
  - TrustMark : seg2 + rectangles 0,743 (n = 27) ; la moyenne n'apporte rien ;
  - StegaStamp : moyenne 0,857 contre 0,848 (gain non significatif, IC [−0,001 ; +0,021]).
- **Limite honnête** : les premiers runs seg2 avaient vu des images impaires, donc la validation commune tombe à 23-27 images pour deux outils (IC ± 0,07). Des runs refaits dans la convention commune (aucune impaire vue) étaient en cours [À COMPLÉTER : résultat `07_t1_seg3` et soumission finale assemblée ou non].

**Q11. "What's the cost at inference?"**
- Segmenteur : un passage avant par image (plus la moyenne de quelques réseaux).
- Recherche de rectangle : coûteuse, mesurée sur A100 à 18 s/image (TrustMark), 27 s (PixelSeal), 58 s (StegaStamp). Acceptable pour de l'audit, pas pour du filtrage à l'échelle d'Internet. C'est une raison de plus pour ne l'appliquer qu'aux images partielles de TrustMark et PixelSeal.

---

## Task 2 : model equality testing

**Résumé de la méthode (pipeline v2, `fingerprint.py`) :**
1. **Aucun canal latéral** : vérifié sur les ref, l'API ne renvoie ni `logprobs` ni `usage`, `finish_reason` vaut toujours `stop`, la latence est identique partout (~27 s), `system` est refusé, `temperature` et `seed` sont ignorés. **Seul le texte est exploitable.**
2. **Familles** :
   - noyaux = endpoints qui se donnent le même nom (« SmolLM … Hugging Face », « Qwen … Alibaba Cloud ») ;
   - **empreinte du tokenizer** (`tokprobe.py`) : le serveur refuse les entrées de plus de 1 024 tokens, comptés avec le tokenizer de l'endpoint. On envoie k répétitions d'un texte et on cherche la frontière : 51 / 35 répétitions pour SmolLM2, 53 / 40 pour Qwen2.5. Les refus sont gratuits : **1 réponse par texte et par endpoint**, et c'est **insensible aux déguisements** (les 9 ref donnent la même frontière) ;
   - les autres endpoints rejoignent la famille qui répond le plus comme eux.
3. **Modèles dans une famille** :
   - nom absent des réponses d'identité → system prompt ou persona → **modèle de base** ;
   - deux répétitions identiques sur ≥ 50 % des prompts → greedy ou beam → **modèle de base** ;
   - sinon, test apparié par prompt sur 5 traits de style (longueur, phrases, « ! », « ? », sauts de ligne) → statistique z ; clustering (average linkage) coupé à z = 3,0.
4. **Prompts** : 50 prompts × 2 répétitions = 100 réponses, en blocs équilibrés (identité, courts, longs) pour pouvoir dépenser le budget par étapes.

**Résultats :**

| Jeu | MCC famille | MCC modèle | Score |
|---|---|---|---|
| ref-01..09 (1 famille, étiqueté) | 0 (1 seule famille) | 0,524 | **0,472** |
| Simulateur local (Qwen2.5-0.5B + SmolLM2-360M, 18 endpoints) | 1,000 | 0,734 | **0,760** |
| ep-01..33 (public) | | | [À COMPLÉTER] |

- Bien classés sur ref : system prompt, persona, watermark, T = 1,0 (rattachés à la base) ; LoRA 0,7 et 1,5 séparés.
- Ratés : **fp8 et LoRA 0,3** fusionnés avec la base.
- Collecte `ep` : 50 réponses par endpoint, lancée sam. 23:16 sur le Mac d'Amil avec `prompts_ref_half.jsonl` [À VÉRIFIER : fichier de prompts réellement utilisé, budget restant, CSV soumis].

**Q12. "Your first approach failed. What happened?"** (bonne histoire à raconter)
- v1 = un quiz de 16 questions courtes par requête, pour récolter 16 « bits » de signature par réponse.
- Sur les ref, le modèle est un **SmolLM** trop petit pour suivre le format (réponses décalées, question recopiée) : AUC 0,25, score 0.
- On l'a vu **gratuitement** sur les ref, avant de dépenser le moindre `ep`. C'est exactement pourquoi on calibre sur les ref.

**Q12b. "Why build your own simulator when you had reference endpoints?"**
- Les 9 ref ne couvrent qu'**une famille** : MCC famille vaut 0 par définition, et c'est trop peu pour régler des seuils.
- Le simulateur (2 familles, greedy, beam, int4, LoRA entraîné) vérifie que les règles **généralisent** à un autre modèle de base.

**Q12c. "How do you distinguish a LoRA at strength 0.3 from a system prompt change?"**
- Le system prompt et la persona **effacent ou changent le nom** que le modèle se donne : c'est notre règle « nom absent → même modèle ».
- La température et le watermark déplacent peu les traits de style (z ≤ 2,2 contre la base à 100 réponses) ; un LoRA ou une quantization 4 bits les déplacent (z = 3,7 à 8,7).
- **Honnêtement** : un LoRA 0,3 reste invisible à notre budget, comme fp8. C'est notre plafond.

**Q13. "How did you spend your 100 queries per endpoint?"**
- Tout le réglage d'abord sur les ref (gratuits et illimités), puis le simulateur.
- **Les mêmes prompts pour tous les endpoints**, dans le même ordre : le test est apparié par prompt.
- Par étapes : passe 0 (empreinte du tokenizer, 2 réponses), passe 1 (50 réponses), puis passe 2 seulement sur les familles ambiguës.
- Collecteur robuste (`test_collect_mock.py`) : reprise après crash sans doublon, aucune réponse décomptée perdue, arrêt propre.

**Q13b. "Some teams got 1.00. Why didn't you?"**
- [À COMPLÉTER selon le score final]
- Ce qu'on sait : aucun canal latéral dans les réponses (pas de logprobs, pas de usage, latence plate). Le signal est forcément dans le texte. Notre faiblesse mesurée : fp8 et LoRA faible.

### Task 2 : questions techniques

**T2.1. "Why not the MMD with Hamming kernel from the paper?"**
- On l'a testé au simulateur : les similarités de texte (sac de mots, trigrammes, Hamming) **ne séparent ni les familles ni les modèles** sur ces petits modèles à 100 réponses.
- Rappel de ce que c'est : noyau = nombre de positions où deux complétions ont le même token ; MMD² = E[k(X,X')] + E[k(Y,Y')] − 2 E[k(X,Y)] ; p-value par permutation.
- Notre z apparié par prompt garde l'idée du papier (comparer des distributions de complétions à prompt fixé) mais sur des **traits agrégés**, plus puissants à petit budget.

**T2.2. "Greedy or beam endpoints are deterministic: won't a distribution test reject them against the sampling base?"**
- Oui, c'est **le piège principal** : une sortie greedy est un Dirac.
- Notre parade : on détecte le déterminisme (≥ 50 % de répétitions identiques) et on le rattache directement au modèle de base, sans passer par le test.

**T2.3. "How do you resist the text watermark?"**
- Un watermark de type Kirchenbauer ajoute δ aux logits des tokens « verts » : il change le choix des mots, peu la longueur ou la ponctuation. Mesuré : z ≤ 2,2 contre la base.
- Observation non utilisée : le watermark des ref a un taux de répétitions identiques de 0,23 (0,03-0,09 pour les autres).

**T2.4. "Why is the tokenizer probe robust?"**
- La limite de 1 024 tokens est appliquée par le **serveur** sur le message brut, avant le modèle : ni system prompt caché, ni persona, ni LoRA, ni quantization ne la changent.
- Elle distingue des tokenizers différents, donc des familles différentes. Deux tailles du même modèle partagent un tokenizer : là, on sépare par le nom.

**T2.5. "528 pairwise tests: how do you keep the answer consistent?"**
- « Même modèle » est une relation d'équivalence. Des tests par paires peuvent se contredire (A = B, B = C, A ≠ C).
- On fait un **clustering** (average linkage sur z) : les groupes sont cohérents par construction.
- Le seuil z se transfère d'un modèle à l'autre : sous « mêmes poids », z est le maximum de 10 statistiques de Student et reste vers 2 (mesuré ≤ 2,2 sur ref, ≤ 2,0 au simulateur). Sensibilité : z = 2,5 à 3,5 donnent le même score sur ref.

**T2.6. "How much statistical power do 100 samples give you?"**
- z croît comme √n. Un LoRA 0,7 est à z = 3,7 avec 100 réponses, donc sous le seuil une fois sur deux avec 50.
- Courbe mesurée : **pas de plateau avant 80 réponses** (simulateur : 0,48 à 50, 0,68 à 80, 0,76 à 100).

**T2.7. "Why MCC and not accuracy?"**
- Les paires « même modèle » sont rares : « tout différent » donnerait une accuracy élevée et un MCC nul.
- MCC modèle n'est calculé que sur les paires d'une même vraie famille (90 % du score) : une erreur de famille se paie deux fois.

**T2.8. "Can't you just ask the model 'which model are you'?"**
- On le fait, mais **pour les familles seulement** (10 % du score) : un SmolLM répond « SmolLM … Hugging Face ».
- Ça ne sépare pas les variantes : une LoRA ou une quantization ne changent pas l'auto-description, et un system prompt peut la masquer. On s'en sert justement à l'envers : nom absent = system prompt ou persona.
- Risque mesuré au simulateur : un LoRA fort peut faire perdre son nom au modèle et passer pour un system prompt.

---

## Task 3 : unlearning du ViT-B/16

**Résumé de notre méthode :**
1. **Pseudo-labels** = prédictions d'`orig` sur le forget set (il les a mémorisées : confiance 0,9997).
2. **Proxy de gold.** On fait passer à `orig` des **vues augmentées** de chaque image (medium, strong, xstrong) : il les reconnaît moins, c'est une approximation de « jamais vue ». Hypothèse retenue : **mixcls** = strong + suppression des 9 classes « plafonnées » à 130 images.
3. **Scoring local** qui reproduit les formules officielles (`scoring.py`) : `mean_std_score` × `prediction_score` × retain.
4. **Target distillation (TD)**, sur **les 12 blocs** :
   - forget : KL à T = 2 vers la cible gold-proxy, + terme `w-stat` qui aligne la moyenne et l'écart-type de la CE du batch sur la cible (le terme qui fait gagner le plus) ;
   - retain : α · KL(orig ‖ sub) sur **Imagenette** ;
   - vues légères augmentées pour couvrir les copies altérées cachées.
5. **Meilleure config** (`sub_lg_g0`) : mixcls, α = 3, lr 5e-5, w-stat 0,1, 8000 pas (meilleur checkpoint à 7000), lot 64 + 64. **0,3045 public.**

**Leviers mesurés au public :**

| Changement | Public |
|---|---|
| cible medium (oubli doux) | 0,0798 |
| cible strong | 0,2384 |
| cible mixcls, α = 10 | 0,2616 |
| α = 5 → 3 → 2 | 0,2736 → 0,2806 → 0,2893 |
| 4000 pas, α = 3 | 0,3043 |
| **8000 pas, α = 3** | **0,3045** (plateau) |

**Q14. "You don't have the gold model: how do you know what to aim for?"**
- On formule **plusieurs hypothèses globales** sur gold (medium, strong, xstrong, suppression des 9 classes, mélanges) et on entraîne une cible pour chacune.
- Le leaderboard sert à **choisir entre ces hypothèses**, pas à régler des paramètres image par image : strong ≫ medium, et mixcls > strong.
- On a ensuite ajusté une famille continue d'hypothèses sur 12-13 points publics (`fit12.py`) : RMS 0,005, LOO 0,008.
- **Puis on l'a testé hors de la zone explorée, et il a échoué** :
  - sonde « suppression totale des 9 classes » (biais de la tête −4) : prédit 0,20-0,22, mesuré 0,2348 → hypothèse « classes entièrement retirées » écartée ;
  - cible ajustée `fitB` : prédit 0,335, mesuré **0,2827** (sous notre meilleur).
- Conclusion qu'on assume : l'ajustement ne vaut qu'**à l'intérieur de la recette mixcls**. On garde `lg_g0` et on ne choisit plus de candidats avec le fit.

**Q15. "Why not just do SCRUB / gradient ascent until forget accuracy is 0?"**
- **Parce que la métrique punit le sur-oubli.** Gold, ré-entraîné sans ces images, les traite comme des images de test ordinaires, pas comme des erreurs systématiques.
- Mesuré : NegGrad+ et SCRUB donnent **0** au score proxy (CE 5 à 58 : distribution détruite). Notre smoke test SCRUB : forget accuracy → 0 et MIA à 0,985, donc un oubli *détectable*.

**Q16. "Why fine-tune the whole model and not just the head?"**
- On a commencé par les dernières couches sur features en cache (CPU, rapide) : bons scores proxy, mais retain dégradé (KL 0,06-0,13).
- Le **modèle complet** (12 blocs) atteint les mêmes scores avec une **KL retain 2 à 4 fois plus faible** (0,01-0,05). C'est la version soumise.
- La version « fin du réseau » reste utile pour tester vite (`tail_ft.py`, exacte par rapport au modèle complet).

**Q17. "What retain data did you use, given none was provided?"**
- **Imagenette** (sous-ensemble public d'ImageNet), 3000 images d'entraînement, 1000 de validation.
- Proxy **pessimiste** : hors distribution, `orig` y hésite déjà, donc l'accord bouge facilement. Accord de `lg_g0` : 0,861, KL 0,047.
- Limite : le vrai control set est in-distribution. Une variante « CutMix d'images forget » a été préparée (`--retain cutmix`) [À VÉRIFIER : résultat de `fa_g3`].

**Q18. "How sure are you to pass the 70 % retain gate?"**
- On ne peut pas le mesurer : pas de données in-distribution.
- Indices : accord 0,86 avec `orig` sur Imagenette, où `orig` est peu sûr ; le public est non nul (un modèle sous le seuil aurait 0).
- Vérifié : **l'évaluation fp16 du serveur ne change rien** (Δlogit max 0,024, aucune classe prédite ne change).

**Q19. "Would this survive a membership inference attack?"**
- C'est l'objectif même de viser gold : si les pertes sur le forget set ressemblent à celles d'images jamais vues, une MIA par la loss ne distingue plus rien. Le terme `w-stat` reproduit justement la queue lourde de pertes de gold.
- Pas de MIA indépendante mesurée sur la soumission finale, on le dit.

**Q20. "Didn't 14 submissions turn the leaderboard into an oracle?"**
- Le public ne porte que sur 30 % du forget set, le classement final sur les 70 % restants.
- Nos choix sont **globaux** (une cible, α, nombre de pas), jamais image par image. Variance de seed mesurée : ±0,005 au proxy.
- Et on a montré nous-mêmes les limites du fit (Q14) au lieu de pousser un candidat « prédit » à 0,335.

### Task 3 : questions techniques

**T3.1. "Your forget score is also evaluated on altered copies you never see. How do you handle that?"**
- Le score oubli est mesuré sur les images à oublier **et** sur des copies altérées cachées, à poids égal.
- On entraîne avec 50 % de vues légèrement augmentées (crop, couleur, flou, `augs.py`) pour que l'oubli **généralise au voisinage** de chaque image.

**T3.2. "Why is 'orig on augmented views' a reasonable proxy for gold?"**
- La mémorisation est **très locale** : `orig` résiste aux augmentations légères (acc 0,99) mais chute avec des fortes (strong : acc 0,52, CE 2,95 ± 4,02).
- Une vue fortement augmentée se comporte comme une image proche jamais vue.
- `p_gold` est la moyenne des softmax de plusieurs modèles gold ; moyenner sur K vues produit le même lissage.
- Limite : le fit suggère un gold **bimodal** (majorité d'images faciles + queue très dure), que nos proxies n'imitent qu'à un facteur d'échelle près (~0,5-0,6).

**T3.3. "How can a model treat forget images differently from unseen images of the same class without labels?"**
- Parce qu'il les **reconnaît** : loss anormalement basse, confiance extrême.
- On corrige ce surplus propre aux exemples mémorisés. Un adoucissement global (diviser tous les logits) fait 0,109 au mieux au proxy : insuffisant.

**T3.4. "Why match both mean AND std of the loss?"**
- La métrique compare moyenne **et** écart-type de la CE avec gold. Gold a une queue lourde (quelques images très fausses).
- La KL seule ne produit pas cette queue ; `w-stat` la crée. C'est le terme qui a le plus fait gagner (proxy medium 0,061 → 0,402).

**T3.5. "Why temperature 2 in the KL?"**
- C'est la définition officielle. T = 2 aplatit les distributions et donne du poids aux classes secondaires : le modèle doit retrouver **quelle mauvaise classe** gold choisirait.

**T3.6. "The 9 'capped' classes: what did you conclude?"**
- 9 classes ont exactement 130 images dans le forget set, un plafond : on a suspecté qu'elles étaient entièrement retirées.
- Le public a d'abord dit oui (mixcls > strong), puis la sonde de biais a dit non à la suppression **totale** (0,2348 < 0,3045). Lecture : gold les traite un peu à part, pas comme des classes jamais vues.

**T3.7. "Pseudo-labels from orig: what if orig is wrong?"**
- `orig` a vu ces images : CE vs pseudo-label 0,0003. Les pseudo-labels ≈ vrais labels.

**T3.8. "Is this certified unlearning?"**
- **Non**, c'est de l'unlearning **approximatif** et empirique, sans garantie formelle.
- Notre critère est l'**indistinguabilité statistique** avec gold, mesurée par la métrique officielle.

**T3.9. "Did weight averaging (model soups) help?"**
- Non, mesuré : soupe greedy 0,507 contre 0,506 pour le meilleur run seul. Extrapolation θ = orig + λ(lg_g0 − orig) : pas de gain prédit, et λ > 1 dégrade le retain.

---

## Questions techniques Task 1 (niveau expert)

**T1.1. "What is the trivial score, and why does the regime decision matter so much?"**
- mIoU **par image, sur 2 classes**, une classe à union vide compte pour 1 (`wmloc/utils.py`).
- Image à petite zone (~4 %) : tout à 0 donne ≈ 0,48 ; image presque entière : tout à 1 donne ≈ 0,47. Plancher ~0,47.
- D'où la règle **jamais de grille vide** : sans aucune case positive, l'image plafonne à 0,5. L'ancienne soumission en avait 434 sur 1020.

**T1.2. "Why score boxes with mean(tanh(logit)·(2b−1)) instead of bit accuracy?"**
- La bit accuracy est discrète (plateaux) : la descente par coordonnées y stagne.
- Le score « soft » est continu, pondère chaque bit par la confiance, et `tanh` borne les logits extrêmes.

**T1.3. "You test thousands of boxes per image: what about false positives?"**
- Sous H0, l'écart-type de la bit accuracy est 0,5/√n : 0,031 pour 256 bits, 0,05 pour 100 bits. Le max sur ~10⁴ boîtes est vers +4σ (≈ 0,62 / 0,70) ; une vraie boîte donne 0,9-1,0.
- Le seuil est réglé **après recherche** sur des images de validation, donc il intègre ce biais de sélection. Par outil : TrustMark 0,60, PixelSeal 0,35. StegaStamp : plancher de bruit trop haut, pas utilisé.

**T1.4. "Why roi_align with adaptive sampling?"**
- Les décodeurs officiels redimensionnent en `bilinear, antialias=True`. Sans anti-aliasing, le repliement de spectre casse le signal haute fréquence.
- `roi_align(sampling_ratio=-1)` moyenne plusieurs points par pixel de sortie : même effet, et des milliers de crops en un seul batch GPU.

**T1.5. "Which loss for a per-image mIoU metric, with tiny positive regions?"**
- Entropie croisée + **mIoU douce** calculée par image, comme la métrique. La BCE seule prédit tout à 0 sur les petites zones.
- Le seuil de binarisation est réglé par régime en validation (la sortie n'est pas une probabilité calibrée).

**T1.6. "Why a head on the frozen PixelSeal detector instead of training from scratch?"**
- Le détecteur officiel sait déjà extraire le signal PixelSeal malgré la modulation JND. Un U-Net from scratch n'y arrive pas (0,27).
- On ajoute l'**accord par position avec le message connu** comme entrée : c'est l'information « message » qu'un segmenteur générique n'a pas.
- Deux variantes indépendantes arrivent au même constat : tête sur le détecteur gelé (Modal, 0,668) et extracteur affiné (JURECA, 0,698). Partir du modèle officiel est la clé pour PixelSeal.

**T1.8. "How do you know the organizers embedded in a shifted frame?"**
- Deux mesures indépendantes concordent :
  - balayage du décodeur TrustMark : il relit le mieux les images officielles dans un cadre 5-10 % plus grand que l'image, alors que nos propres marquages culminent au cadre exact ;
  - ré-embarquer le même message sur une image officielle ajoute autant de résidu que sur une image propre (1,34 contre 0,75 attendu) ; un décalage de 8 px ou un zoom à 90 % reproduit cet écart.
- On fabrique donc les exemples dans un cadre agrandi aléatoire (0 à 32 px par côté, remplissage miroir).

**T1.7. "Your box is pixel-precise but the metric uses 4×4 cells: how do you convert?"**
- Même règle que l'officiel : boîte dessinée à 512 px, moyennée par cellules 4 × 4, seuil ≥ 0,5. Petit agrandissement `d` (0 à 3 px) réglé en validation, parce que la recherche rogne les bords.

---

## Pièges de présentation
- **Ne pas survendre** : dire clairement ce qui ne marche pas (PixelSeal sur les petits trous, fp8 et LoRA 0,3 en T2, plafond de T3 et échec du fit hors famille).
- **Une slide par tâche** : problème → idée clé → résultat validé → limite.
- **Idées clés à mettre en avant :**
  - T1 : on a **rétro-conçu la recette des organisateurs** (crop → embed → paste pour les rectangles, masques SAM pour les formes, cadre TrustMark décalé), puis fabriqué des données fidèles. Progression publique 0,26 → 0,77, validation locale à ±0,002 du public.
  - T2 : l'**empreinte du tokenizer** par la limite de 1 024 tokens, gratuite et insensible aux déguisements. Et l'échec du quiz détecté gratuitement sur les ref.
  - T3 : viser gold, pas zéro ; `w-stat` pour reproduire la queue de pertes ; et l'honnêteté sur le fit (prédit 0,335, mesuré 0,283).
