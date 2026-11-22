# Préparation aux questions du jury

_Mis à jour le dim. 22/11 (~02:35, `main` + `develop`). Sources : `challenge/task1_modal_report.md`, `challenge/task1_watermark_report.md`, `challenge/COORD_T1.md`, `challenge/task2_model_equality_report.md`, `challenge/task3_unlearning_report.md`._

## Mode d'emploi

- Le jury (CISPA / SprintML, Meta) interroge les 6 meilleures équipes. **La note finale combine les scores et la qualité des explications.**
- Le jury pose ses questions en anglais. Elles sont écrites en anglais ici, avec des réponses en français : à dire en anglais.
- Pour chaque question :
  - **À dire d'abord** = une phrase de réponse. Si le jury n'en demande pas plus, on s'arrête là ;
  - les points en dessous = les détails, si le jury creuse.
- **Qui répond** : chaque partie a un responsable principal et un remplaçant. Tout le monde doit quand même connaître la partie 0 et la « fiche 30 secondes » de chaque tâche.
- `[À COMPLÉTER]` = chiffre à récupérer avant le pitch (liste à la fin).

| Partie | Contenu | Répond en premier | Remplaçant |
|---|---|---|---|
| 0 | Questions générales, toute l'équipe | [À COMPLÉTER] | tous |
| 1 | Task 1 : localisation de watermark | [À COMPLÉTER : Amil et/ou Matis ?] | |
| 2 | Task 2 : model equality testing | [À COMPLÉTER : Amil ?] | |
| 3 | Task 3 : unlearning | [À COMPLÉTER : Matis ?] | |

---

# Partie 0 : questions générales

## Nos chiffres, à connaître par cœur

| Tâche | Score public | Évolution | Validation locale |
|---|---|---|---|
| T1 watermark | **0,7656** (v4c : [À COMPLÉTER]) | 0,26 → 0,46 → 0,71 → 0,75 → 0,77 | 0,784 pour v4c ; le local prédit le public à ±0,002 |
| T2 model equality | [À COMPLÉTER] | 0,00 → ? | 0,472 sur les endpoints de référence, 0,760 sur notre simulateur |
| T3 unlearning | **0,3045** | 0,08 → 0,24 → 0,26 → 0,29 → 0,30 | proxy : accord retain 0,861 |

## G1. "Is this solution really yours? Walk us through this line."
**À dire d'abord :** « Oui. Chaque choix vient d'une expérience qu'on peut montrer, y compris celles qui ont échoué. »
- Fichiers que le responsable de chaque tâche doit savoir expliquer :
  - T1 : `modal_t1/train_lib.py` (U-Net), `modal_t1/pixelseal_research/det_head.py` (tête PixelSeal), `solution/train_seg2.py` (extracteur affiné), `solution/rectsearch.py` (recherche de rectangle), `solution/ensemble.py` (assemblage) ;
  - T2 : `fingerprint.py` (familles et modèles), `tokprobe.py` (empreinte du tokenizer), `collect.py` (collecte) ;
  - T3 : `unlearn.py` (entraînement), `scoring.py` (score local), `fit12.py` (hypothèses sur gold).
- Si on nous demande si on a utilisé un assistant de code : le dire honnêtement, puis montrer qu'on comprend chaque choix.

## G2. "How do you know you didn't overfit the public leaderboard?"
**À dire d'abord :** « Sur la Task 1, notre validation locale prédit le score public à ±0,002. Sur la Task 3, on a mesuré nous-mêmes les limites de notre ajustement et on a arrêté de l'utiliser. »
- T1 : validation sur la moitié impaire des images officielles, jamais utilisée pour l'entraînement. Local 0,705 → public 0,7065 ; 0,750 → 0,7523 ; 0,767 → 0,7656.
- T2 : réglages faits sur les endpoints de référence et notre simulateur. **Pas de jeu de validation indépendant**, on le dit.
- T3 : voir partie 3, question U5.

## G3. "What would you do with more time or compute?"
**À dire d'abord :** « Chaque tâche a une faiblesse identifiée et mesurée ; c'est là qu'on mettrait le temps. »
- T1 : les images presque entièrement marquées avec un petit trou propre (45 % de nos points perdus).
- T2 : dépenser le reste du budget sur les cas ambigus ; notre score monte encore jusqu'à 80 réponses par endpoint.
- T3 : des données « à garder » proches de la vraie distribution, et une information par image sur le modèle gold.

## G4. "You used paid cloud compute (Modal). Is your result reproducible on the provided cluster?"
**À dire d'abord :** « Oui, le calcul tient sur des A100. On a utilisé Modal parce que la règle limite chaque équipe à un seul job sur JURECA. »
- Sur Modal : 2 × 25 000 exemples fabriqués par outil, un U-Net par outil (H100, 12 000 pas), jusqu'à ~10 GPU en parallèle.
- Sur JURECA : l'extracteur affiné de la Task 1 (70-75 min sur A100) et tout l'entraînement de la Task 3.
- [À COMPLÉTER : temps total sur Modal et équivalent A100]

## G5. "What is the link with the AI Act?"
**À dire d'abord :** « Les trois tâches sont trois outils d'audit prévus par la régulation. »
- **Art. 50**, marquage des contenus générés → T1 : savoir *où* une image est marquée.
- **Art. 53**, transparence des fournisseurs → T2 : vérifier qu'une API sert le modèle annoncé.
- **RGPD, droit à l'effacement** → T3 : retirer des données d'un modèle sans le ré-entraîner.

## Pièges à éviter (toute l'équipe)
- **Ne pas survendre.** Dire ce qui ne marche pas : PixelSeal sur les petits trous (T1), fp8 et LoRA faible (T2), plafond et échec de l'ajustement (T3).
- Ne pas inventer un chiffre : répondre « on ne l'a pas mesuré ».
- Une slide par tâche : problème → idée clé → résultat validé → limite.

---

# Partie 1 : Task 1, localisation de watermark

## Fiche 30 secondes (tout le monde)
On reçoit une image et l'outil de watermark utilisé (PixelSeal, TrustMark ou StegaStamp). Il faut dire, sur une grille 128 × 128, quelles zones sont marquées.
1. On a **retrouvé la recette des organisateurs** (comment ils ont marqué et collé les zones).
2. On a **fabriqué des exemples fidèles** avec les outils officiels.
3. On a entraîné **un segmenteur par outil**, en partant des modèles officiels quand c'est nécessaire (PixelSeal).
4. Résultat : **0,26 → 0,77 public**, avec une validation locale fiable à ±0,002.

## Chiffres clés

| Version | PixelSeal | TrustMark | StegaStamp | Moyenne locale | Public |
|---|---|---|---|---|---|
| Baseline (décision globale + seuillage) | 0,479 | 0,459 | 0,460 | 0,466 | 0,4584 |
| v1 : U-Net + rectangles | 0,571 | 0,692 | 0,852 | 0,705 | 0,7065 |
| v3a | | | | 0,767 | 0,7656 |
| v4c : réseaux plus gros, image x2, deux séries croisées | 0,668 | 0,759 | 0,925 | **0,784** | [À COMPLÉTER] |
| Pipeline JURECA (extracteur affiné), seul | **0,698** | 0,743 | 0,822 | — | non soumis |

## La méthode en 6 étapes
1. **Décision globale.** On décode le message sur l'image entière : moins de 80 % de bits relus = image « partielle » (petite zone marquée), sinon « complète ». Seulement 3 erreurs sur 271 images.
2. **Recette des organisateurs**, retrouvée par diagnostic :
   - rectangles et barres = morceau **découpé, marqué, recollé** ;
   - formes irrégulières des images partielles = **masques SAM de l'image** (IoU médian 0,91) ;
   - trous des images complètes = taches lisses (64 %) ou masques rugueux (36 %) ;
   - TrustMark marqué dans un **cadre 5-10 % plus grand** que l'image.
3. **Exemples fabriqués** avec les outils du kit officiel : 25 000 sur COCO + 25 000 à partir des images officielles paires. Le mélange bat chacun seul.
4. **Segmenteurs** :
   - TrustMark, StegaStamp : U-Net EfficientNet-B4/B6, entrée = image + micro-détails (image − flou, × 20), image agrandie x2 ;
   - PixelSeal : tête sur le détecteur officiel (Modal, 0,668) ou extracteur officiel affiné (JURECA, 0,698), avec l'accord par position avec le message connu.
5. **Recherche de rectangle** en complément : on décode chaque boîte candidate et on garde celle qui relit le mieux le message (TrustMark et PixelSeal seulement).
6. **Assemblage** de plusieurs réseaux, et jamais de grille vide ni pleine.

## Questions de base

**W1. "Why does decoding the exact crop work? Isn't that exploiting a dataset artifact?"**
**À dire d'abord :** « C'est une propriété du marquage : un watermark posé sur un morceau d'image n'est lisible qu'à la même échelle et au même cadrage. »
- Sur la boîte exacte, 90-100 % de bits corrects, contre ~50 % ailleurs (`croptest.py`, `basin.py`).
- Tolérance au décalage : ±8-16 px pour TrustMark, ±2-4 px pour PixelSeal et StegaStamp.
- C'est ce que ferait un vrai auditeur : il connaît son message et cherche où il est lisible.
- Ce n'est **plus notre méthode principale** : elle ne couvre que les rectangles (25 % des images). TrustMark l'utilise sur 24 rectangles sur 26 ; pour PixelSeal, elle n'apporte presque rien.

**W2. "Why a synthetic dataset, and how do you make sure it matches the real one?"**
**À dire d'abord :** « 510 images réelles ne suffisent pas. On a reproduit la recette mesurée avec les outils officiels, et on valide uniquement sur les vraies images. »
- Mêmes versions des outils que les organisateurs (`trustmark`, `torch`) : une autre version changerait le motif.
- Contrôle de fidélité : la trace TrustMark officielle était 6 fois moins corrélée qu'en simulation. Cause trouvée : le cadre décalé (W8). On fabrique donc dans un cadre agrandi aléatoire.

**W3. "Is it realistic to assume the message is known?"**
**À dire d'abord :** « Oui : le cas d'usage principal est un fournisseur qui vérifie son propre watermark. »
- Il connaît sa clé et son message (Meta, Google SynthID…). L'énoncé fournit d'ailleurs le message.

**W4. "Why is PixelSeal harder than the others?"**
**À dire d'abord :** « PixelSeal cache son signal là où l'œil ne le voit pas, donc presque rien dans les zones plates. »
- Il module la force par une carte de visibilité (JND) : signal quasi nul sur le ciel ou un mur.
- Un U-Net seul n'apprend rien (0,27). Il faut partir du **détecteur officiel**, qui sait déjà où lire.
- Les rectangles fins (un côté < 20-30 px) sont illisibles même sur la boîte exacte.

**W5. "Why is StegaStamp your best tool?"**
**À dire d'abord :** « Sa trace est forte et le U-Net l'apprend bien : 0,925. »
- La recherche de rectangle, en revanche, échoue (bruit trop haut, 1 rectangle trouvé sur 23) : on ne l'utilise pas.

**W6. "Where do you still lose points?"**
**À dire d'abord :** « Sur les images presque entièrement marquées avec un petit trou propre, surtout pour PixelSeal et TrustMark : 45 % de nos pertes. »
- Plus le trou est petit, plus le score baisse (PixelSeal 0,65 pour 10-30 % de trou, 0,48 sous 2 %).
- Le seuil n'y est pour rien (+2 points au mieux) : c'est la carte qui manque de signal.

**W7. "Two sub-teams worked on Task 1. How did you combine them?"**
**À dire d'abord :** « Même format de sortie, même validation, et un banc qui ne garde une combinaison que si le gain est statistiquement solide. »
- Pipelines : U-Net sur Modal ; extracteur officiel affiné sur JURECA (`train_seg2.py`).
- `ensemble.py` : validation croisée 10 × 5 plis, gain minimal 0,005 avec probabilité bootstrap ≥ 0,80.
- Résultat : la moyenne paie sur PixelSeal (0,707 contre 0,698, méthodes différentes), pas sur TrustMark.
- Limite : certains runs avaient vu des images de validation, donc seulement 23-27 images propres pour deux outils. Des runs refaits proprement tournaient [À COMPLÉTER : résultat et fichier final].

## Questions expertes

**W8. "How do you know the organizers embedded in a shifted frame?"**
- Deux mesures indépendantes :
  - le décodeur TrustMark relit le mieux les images officielles dans un cadre 5-10 % plus grand, alors que nos marquages culminent au cadre exact ;
  - ré-embarquer le message sur une image officielle ajoute autant de résidu que sur une image propre (1,34 contre 0,75) ; un décalage de 8 px ou un zoom à 90 % reproduit l'écart.

**W9. "What is the trivial score, and why does the global decision matter?"**
- mIoU par image sur 2 classes. Tout prédire à 0 sur une image à petite zone donne ≈ 0,48 : le plancher est ~0,47.
- Une grille vide plafonne l'image à 0,5. Notre première soumission en avait 434 sur 1020 → règle « jamais de grille vide ».

**W10. "Why upsample x2? Why the high-pass channel?"**
- Le watermark est faible et haute fréquence : x2 donne plus de pixels de réseau par motif. C'est le plus gros levier (TrustMark 0,697 → 0,738 ; StegaStamp 0,893 → 0,926).
- Sans canal micro-détails, le calcul en bf16 efface la trace par arrondi.

**W11. "Why no flip or rotation augmentation?"**
- Un watermark retourné n'est plus un watermark valide. On n'augmente que l'image *avant* marquage.

**W12. "Which loss, given tiny positive regions?"**
- Entropie croisée + mIoU douce par image, comme la métrique. Le seuil est réglé par régime en validation.

**W13. "You test thousands of boxes per image: what about false positives?"**
- Sans watermark, la bit accuracy a un écart-type de 0,031 (256 bits) ou 0,05 (100 bits). Le maximum sur ~10⁴ boîtes est vers 0,62 / 0,70 ; une vraie boîte donne 0,9-1,0.
- Le seuil est réglé après recherche (TrustMark 0,60 ; PixelSeal 0,35), donc il intègre ce biais.

**W14. "Why score boxes with mean(tanh(logit)·(2b−1)) and roi_align?"**
- Le score doux est continu (la bit accuracy fait des plateaux) et `tanh` borne les logits extrêmes.
- `roi_align` reproduit le redimensionnement anti-aliasé des décodeurs officiels et traite des milliers de crops en un batch.

**W15. "What's the cost at inference?"**
- Segmenteur : un passage par image et par réseau.
- Recherche de rectangle sur A100 : 18 s/image (TrustMark), 27 s (PixelSeal). Acceptable pour de l'audit, pas à l'échelle d'Internet.

---

# Partie 2 : Task 2, model equality testing

## Fiche 30 secondes (tout le monde)
33 endpoints d'API, 100 réponses maximum chacun. Il faut regrouper ceux qui servent le même modèle, malgré des déguisements (system prompt, persona, watermark, température) et de vrais changements (quantization fp8, LoRA).
1. **Aucun canal caché** : pas de probabilités, latence identique. Seul le texte compte.
2. **Familles** : le nom que le modèle se donne, et une **empreinte du tokenizer** obtenue grâce à la limite de 1 024 tokens du serveur.
3. **Modèles** : règles pour les déguisements, puis test statistique sur des traits de style.
4. Résultat : 0,472 sur les endpoints de référence, 0,760 sur notre simulateur, [À COMPLÉTER] en public.

## Chiffres clés

| Jeu | MCC famille | MCC modèle | Score |
|---|---|---|---|
| Endpoints de référence ref-01..09 (1 famille) | 0 (une seule famille) | 0,524 | **0,472** |
| Simulateur (Qwen2.5 + SmolLM2, 18 endpoints) | 1,000 | 0,734 | **0,760** |
| ep-01..33, public | | | [À COMPLÉTER] |

- Bien classés : system prompt, persona, watermark, température (= même modèle) ; LoRA 0,7 et 1,5 (= différents).
- **Ratés : fp8 et LoRA 0,3**, confondus avec le modèle de base.
- Collecte réelle : 50 réponses par endpoint, lancée sam. 23:16 [À VÉRIFIER : fichier de prompts, budget restant, CSV soumis].

## La méthode
1. **Vérifié sur les endpoints de référence** : ni `logprobs` ni `usage`, `finish_reason` toujours `stop`, latence ~27 s partout, `system` refusé, `temperature` et `seed` ignorés.
2. **Familles** :
   - nom que le modèle se donne (« SmolLM … Hugging Face », « Qwen … Alibaba Cloud ») ;
   - **empreinte du tokenizer** : on envoie un texte répété k fois et on cherche à partir de quand le serveur refuse (limite de 1 024 tokens). 51 / 35 répétitions pour SmolLM2, 53 / 40 pour Qwen2.5. Refus gratuits, **1 réponse par texte**, insensible aux déguisements.
3. **Modèles dans une famille** :
   - le nom disparaît → system prompt ou persona → même modèle ;
   - réponses répétées identiques → décodage greedy ou beam → même modèle ;
   - sinon, test apparié sur 5 traits de style (longueur, phrases, « ! », « ? », sauts de ligne) → statistique z, regroupement coupé à z = 3.
4. **Prompts** : 50 prompts × 2 répétitions, en blocs équilibrés, pour dépenser le budget par étapes.

## Questions de base

**M1. "Your first approach failed. What happened?"** (bonne histoire à raconter)
**À dire d'abord :** « Notre premier plan était un quiz ; le modèle de référence était trop petit pour le suivre. On l'a vu gratuitement, avant de dépenser le budget. »
- Quiz de 16 questions par requête, pour récolter 16 indices par réponse.
- Le modèle des endpoints de référence est un SmolLM : réponses décalées, question recopiée. Score 0.
- C'est exactement pour ça qu'on calibre d'abord sur les endpoints de référence, illimités et gratuits.

**M2. "How did you spend your 100 queries per endpoint?"**
**À dire d'abord :** « On a tout réglé gratuitement sur les endpoints de référence, puis envoyé les mêmes prompts à tous les endpoints, par étapes. »
- Mêmes prompts, même ordre : le test compare les endpoints prompt par prompt.
- Étapes : empreinte du tokenizer (2 réponses), 50 réponses, puis le reste seulement sur les familles ambiguës.
- Collecteur testé : reprise après crash sans doublon ni réponse perdue.

**M3. "How do you tell a LoRA at strength 0.3 from a system prompt change?"**
**À dire d'abord :** « Le system prompt change le nom que le modèle se donne ; une LoRA change son style de façon mesurable. Mais à 0,3, honnêtement, on ne la voit pas. »
- Température et watermark : z ≤ 2,2 contre la base. LoRA ou quantization 4 bits : z = 3,7 à 8,7.
- LoRA 0,3 et fp8 restent invisibles à notre budget : c'est notre plafond.

**M4. "Why build your own simulator when you had reference endpoints?"**
**À dire d'abord :** « Les endpoints de référence n'ont qu'une famille : impossible d'y tester la séparation des familles. »
- Simulateur : 2 familles, greedy, beam, int4, LoRA entraîné. Il vérifie que nos règles marchent sur un autre modèle.

**M5. "Can't you just ask the model 'which model are you'?"**
**À dire d'abord :** « On le fait, mais seulement pour les familles. Pour les variantes, on s'en sert à l'envers. »
- Une LoRA ou une quantization ne changent pas le nom ; un system prompt peut le masquer.
- Nom absent = system prompt ou persona = même modèle.
- Risque vu au simulateur : une LoRA forte peut faire perdre son nom au modèle.

**M6. "Some teams got 1.00. Why didn't you?"**
**À dire d'abord :** [À COMPLÉTER selon le score final]
- Ce qu'on sait : aucun canal caché dans les réponses, le signal est dans le texte. Nos faiblesses mesurées : fp8 et LoRA faible.

## Questions expertes

**M7. "Why not the MMD test with a Hamming kernel from the paper?"**
- On l'a testé au simulateur : les similarités de texte (sac de mots, trigrammes, Hamming) ne séparent rien sur ces petits modèles à 100 réponses.
- Rappel : noyau = nombre de positions où deux complétions ont le même token ; MMD² = E[k(X,X')] + E[k(Y,Y')] − 2E[k(X,Y)] ; p-value par permutation.
- Notre test garde l'idée du papier (comparer à prompt fixé) mais sur des traits agrégés, plus puissants à petit budget.

**M8. "Why is the tokenizer probe robust?"**
- La limite est appliquée par le **serveur**, sur le message brut, avant le modèle : aucun déguisement ne la change.
- Deux tailles d'un même modèle partagent un tokenizer : là, on sépare par le nom.

**M9. "Greedy or beam endpoints are deterministic. Won't a distribution test reject them?"**
- Oui, c'est le piège principal. On détecte le déterminisme (≥ 50 % de répétitions identiques) et on le rattache directement à la base.

**M10. "How do you resist the text watermark?"**
- Le watermark favorise certains mots, mais change peu la longueur ou la ponctuation : z ≤ 2,2 contre la base.

**M11. "528 pairwise tests: how do you stay consistent?"**
- Des tests par paires peuvent se contredire (A = B, B = C, A ≠ C). On fait un **regroupement** (average linkage sur z) : cohérent par construction.
- Le seuil se transfère : sous « mêmes poids », z reste vers 2 quel que soit le modèle. z = 2,5 à 3,5 donnent le même score sur ref.

**M12. "How much statistical power do 100 samples give you?"**
- z croît comme √n. LoRA 0,7 : z = 3,7 à 100 réponses, donc ratée une fois sur deux à 50.
- Simulateur : 0,48 à 50 réponses, 0,68 à 80, 0,76 à 100. Pas de plateau avant 80.

**M13. "Why MCC and not accuracy?"**
- Les paires « même modèle » sont rares : « tout différent » donnerait une bonne accuracy et un MCC nul.
- Le score modèle (90 %) ne compte que les paires d'une même famille : une erreur de famille se paie deux fois.

---

# Partie 3 : Task 3, unlearning d'un ViT

## Fiche 30 secondes (tout le monde)
Un classifieur d'images (ViT-B/16) doit « oublier » 3 500 images, comme s'il n'avait jamais été entraîné dessus. Le score compare notre modèle à un modèle « gold » ré-entraîné sans elles, qu'on ne voit pas.
1. **Viser gold, pas zéro** : oublier trop est puni.
2. On imite gold en montrant au modèle des **versions fortement modifiées** des images : il les reconnaît moins, comme des images jamais vues.
3. On entraîne le modèle complet vers cette cible, en restant proche de l'original sur d'autres images.
4. Résultat : **0,3045 public**, en progression régulière depuis 0,08.

## Chiffres clés

| Changement | Public |
|---|---|
| cible « oubli doux » (medium) | 0,0798 |
| cible « oubli fort » (strong) | 0,2384 |
| cible mixcls (strong + 9 classes traitées à part), α = 10 | 0,2616 |
| α = 5 → 3 → 2 (moins de poids sur la préservation) | 0,2736 → 0,2806 → 0,2893 |
| 4000 pas, α = 3 | 0,3043 |
| **8000 pas, α = 3 (`sub_lg_g0`, soumission finale)** | **0,3045** (plateau) |

## La méthode
1. **Pseudo-labels** = prédictions du modèle original sur les images à oublier (il les connaît par cœur : confiance 0,9997).
2. **Imitation de gold** : on fait passer au modèle original des vues fortement modifiées de chaque image. Sa réponse sert de cible.
3. **Score local** (`scoring.py`) qui reproduit les formules officielles avec cette cible.
4. **Entraînement** (« target distillation »), sur les 12 blocs :
   - images à oublier : rapprocher les sorties de la cible, et aligner la moyenne et l'écart-type de l'erreur (terme `w-stat`, celui qui fait le plus gagner) ;
   - images à garder : rester proche du modèle original sur Imagenette (poids α) ;
   - vues légèrement modifiées pour couvrir les copies altérées que le serveur teste.
5. Réglages finaux : α = 3, lr 5e-5, w-stat 0,1, 8000 pas.

## Questions de base

**U1. "You don't have the gold model. How do you know what to aim for?"**
**À dire d'abord :** « On a formulé quelques hypothèses globales sur gold, une cible par hypothèse, et le leaderboard a servi à choisir entre elles. »
- Hypothèses : oubli doux, fort, très fort, 9 classes retirées, mélanges.
- Résultat clair : fort ≫ doux, et mixcls > fort.
- Suite et limites : voir U5.

**U2. "Why not just do gradient ascent or SCRUB until forget accuracy is 0?"**
**À dire d'abord :** « Parce que la métrique punit le sur-oubli : gold ne se trompe pas sur toutes ces images, il les traite comme des images normales. »
- Mesuré : gradient ascent et SCRUB donnent 0 au score local (sorties détruites).
- Smoke test SCRUB : précision → 0, et une attaque d'appartenance détecte l'oubli à 0,985.

**U3. "Why fine-tune the whole model rather than just the last layers?"**
**À dire d'abord :** « Les deux marchent sur l'oubli, mais le modèle complet abîme 2 à 4 fois moins les images à garder. »
- Dernières couches seules : rapide, mais KL sur les images à garder 0,06-0,13.
- Modèle complet : 0,01-0,05. C'est la version soumise.

**U4. "What retain data did you use, given none was provided?"**
**À dire d'abord :** « Imagenette, un sous-ensemble public d'ImageNet. C'est un choix prudent, mais pas la bonne distribution. »
- 3000 images d'entraînement, 1000 de validation. Accord avec l'original : 0,861.
- Le vrai jeu de contrôle est différent. Variantes testées : mélanges d'images à oublier (CutMix), DiffusionDB [À VÉRIFIER : résultats].

**U5. "Didn't 14 submissions turn the leaderboard into an oracle?"**
**À dire d'abord :** « On a ajusté un modèle d'hypothèses sur nos scores publics, on l'a testé, il a échoué hors de notre recette, et on a arrêté de s'en servir. »
- Ajustement sur 12-13 points : erreur 0,005.
- Deux tests hors de la zone explorée :
  - « les 9 classes sont entièrement retirées » : prédit 0,20-0,22, mesuré 0,2348 → hypothèse écartée ;
  - nouvelle cible ajustée : prédit 0,335, **mesuré 0,283**, sous notre meilleur.
- Le classement final porte sur les 70 % d'images non publiques, et nos choix sont globaux (cible, α, pas), jamais image par image. Variance entre graines : ±0,005.

**U6. "How sure are you to pass the 70 % retain gate?"**
**À dire d'abord :** « On ne peut pas le mesurer, mais tous les indices sont bons. »
- Accord 0,86 avec l'original sur Imagenette, où l'original hésite déjà.
- Le score public est non nul : sous le seuil, il serait 0.
- L'évaluation en fp16 du serveur ne change rien (écart de logit max 0,024, aucune prédiction modifiée).

**U7. "Would this survive a membership inference attack?"**
**À dire d'abord :** « C'est l'objectif de viser gold, mais on ne l'a pas mesuré sur la soumission finale. »
- Si les erreurs sur les images oubliées ressemblent à celles d'images jamais vues, une attaque par la loss ne distingue plus rien. `w-stat` vise justement cette distribution.

**U8. "Is this certified unlearning?"**
**À dire d'abord :** « Non, c'est de l'unlearning approximatif, sans garantie formelle. »
- Notre critère est l'indistinguabilité statistique avec gold, mesurée par la métrique officielle.

## Questions expertes

**U9. "Why is 'original model on augmented views' a reasonable proxy for gold?"**
- La mémorisation est très locale : l'original résiste aux petites modifications (précision 0,99) mais chute avec des fortes (0,52).
- Gold est une moyenne de plusieurs modèles ; moyenner sur plusieurs vues donne le même lissage.
- Limite : l'ajustement suggère un gold bimodal (beaucoup d'images faciles, une queue très dure), qu'on n'imite qu'à un facteur d'échelle près (~0,5-0,6).

**U10. "The 9 'capped' classes: what did you conclude?"**
- 9 classes ont exactement 130 images à oublier, un plafond : on a soupçonné qu'elles étaient entièrement retirées.
- Le public a d'abord confirmé (mixcls > fort), puis la sonde a infirmé une suppression **totale**. Gold les traite un peu à part, pas comme des classes jamais vues.

**U11. "Why match both mean AND std of the loss?"**
- La métrique compare les deux. Gold a une queue lourde : quelques images très fausses.
- La KL seule ne la crée pas ; `w-stat` oui (score local 0,061 → 0,402).

**U12. "Your forget score also uses hidden altered copies. How do you handle that?"**
- La moitié des exemples d'entraînement sont des vues légèrement modifiées : l'oubli couvre le voisinage de chaque image.

**U13. "Why temperature 2 in the KL?"**
- C'est la définition officielle. T = 2 donne du poids aux classes secondaires : il faut retrouver **quelle mauvaise classe** gold choisirait.

**U14. "Pseudo-labels from the original model: what if it is wrong?"**
- Il a vu ces images : erreur moyenne 0,0003. Pseudo-labels ≈ vrais labels.

**U15. "Did weight averaging (model soups) help?"**
- Non, mesuré : 0,507 contre 0,506 pour le meilleur run seul. Extrapoler les poids au-delà de notre modèle abîme les images à garder.

---

# À compléter avant le pitch

| Partie | Chiffre manquant | À demander à |
|---|---|---|
| Tableau « qui répond » | responsables et remplaçants | équipe |
| 0 / G4 | temps de calcul Modal et équivalent A100 | Amil |
| 1 | score public de v4c, fichier T1 final (assemblé ou non) | Amil, Matis |
| 1 / W7 | résultat des runs JURECA refaits proprement (`07_t1_seg3`) | Matis |
| 2 | score public T2, fichier de prompts utilisé, budget restant | Amil |
| 2 / M6 | réponse « pourquoi pas 1,00 » selon le score final | Amil |
| 3 / U4 | résultats CutMix (`fa_g3`) et DiffusionDB | Matis |
