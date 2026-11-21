# CONTEXTE AGENT — Talk organisateur : « Invisible image watermarking between research and production » (Pierre Fernandez, Meta, CISPA, 21/11/2026)
Source : deck Slidev https://pierrefdz.github.io/assets/slides/2026-11-21_image_watermarking_cispa.html. Ce sont 57 slides ; le texte et les **notes du speaker** ont été extraits dans `papers/txt/organizer_slides_raw.txt`. Les tableaux et figures des slides sont des images non extraites : les chiffres ci-dessous viennent des notes orales.
**À lire en priorité pour un challenge watermarking : c'est ce que l'organisateur considère important.**

## Plan du talk
1. Pourquoi watermarker → 2. Post-hoc (Pixel Seal, WAM) → 3. Génération (Stable Signature, DistSeal, Tree-Ring, autorégressif) → 4. Problèmes de déploiement.

## 1. Motivation
- Risques de la GenAI : un guide de champignons écrit par une IA vendu sur Amazon a envoyé quelqu'un à l'hôpital ; influence d'opinion, scams, pollution d'internet.
- Les alternatives sont faibles :
  - **forensics / détection de deepfake** : difficile, et de plus en plus avec de meilleurs générateurs ;
  - **métadonnées** (C2PA, Content Authenticity Initiative) : non persistantes, une capture d'écran les efface ;
  - **fingerprinting** : approche aussi présentée dans le talk.
- **Obligations légales** :
  - **AI Act UE** : marquage machine-readable obligatoire à partir d'**août 2026** ;
  - **SB 942 Californie** : watermark + outil public de détection, même date ;
  - **AI Basic Act Corée** : en vigueur depuis janvier.
- **3 critères** : imperceptibilité (faible distorsion), payload (longueur du message ↔ confiance de détection), robustesse (survivre aux édits).
- **Où watermarker** (pipeline données → entraînement → inférence → pixels) :
  - forensics (aucune marque) ;
  - métadonnées ;
  - **post-hoc** sur les pixels ;
  - inférence : bruit initial ou sampling ;
  - entraînement : poids ou décodeur.

## 2. Post-hoc avec réseaux profonds
- **Recette** : embedder (image + message → image marquée) et extractor (image attaquée → message), entraînés conjointement avec une perte perceptuelle + une perte watermark. Des **transformations aléatoires au milieu** donnent la robustesse « gratuitement ».
- **Détection = test d'hypothèse** :
  - on cache un message fixe m′ et on compte les bits égaux M(m, m′) ;
  - H0 : image non générée par notre modèle. Sous H0, les bits sont iid ~ Bernoulli(0,5) : hypothèse « vérifiée en pratique » selon le speaker ;
  - `P(M ≥ τ | H0) = (1/2^n) Σ_{i=τ+1}^{n} C(n, i)` ;
  - exemple de la slide : **17/32 bits → non flaggé ; 31/32 → flaggé**.
- **Évaluation** :
  - TPR = part d'images IA correctement flaggées, à faire tendre vers 1 ;
  - **FPR** = probabilité de flagger une image naturelle. FPR = 0 est impossible ; il faut la **contrôler précisément à des valeurs très basses, ex. 1e-9**, pour ne pas accuser à tort ;
  - le seuil τ fixe le compromis → **courbes ROC**.
- **Lignée** (slide 15) :

  | Méthode | Idée |
  |---|---|
  | HiDDeN | la recette |
  | TrustMark | 256×256 + resize → toute résolution |
  | SynthID-Image | même idée à l'échelle d'internet |
  | Video Seal | même recette pour la vidéo |
  | Pixel Seal | adversarial seul + JND |
  | WAM | localisation |

  Nouvelles capacités : méthodes localisées, détection de forgeries, plus de bits. Toutes partagent la même recette et diffèrent par la résolution, l'imperceptibilité, la stabilité d'entraînement et la localisation.
- **Pixel Seal** (slides 16-22) :
  - pipeline : carte JND → modulation par heatmap perceptuelle ; **pas de pertes perceptuelles** ;
  - schedule en 3 étapes : (1) pas de discriminateur, α = 1,0 ; (2) discriminateur ajouté, α toujours 1,0 ; (3) α descend à 0,2, puis fine-tuning sur des attaques supplémentaires ;
  - « Pixel Seal est le plus robuste, et aucune autre méthode n'est à la fois plus invisible et plus robuste » (frontière de Pareto) ;
  - **open source, licence MIT**, repo videoseal.
- **WAM** (slides 23-34) :
  - **les watermarks globaux ratent les édits locaux** : inpainting (cheval → zèbre), splicing (2 canards marqués par 2 modèles avec 2 messages), crop fort ;
  - solution : **extraction = segmentation** (esprit Segment Anything), avec une carte de détection + un message par pixel ;
  - embedder : architecture de l'autoencodeur Stable Diffusion en plus petit, downsampling ×16. Chaque bit choisit un de 2 vecteurs appris ; on moyenne, on répète sur la grille latente et on ajoute au latent z ;
  - augmenter : crop, resize, rotation, perspective, flip, luminosité, contraste, saturation, teinte, flou, JPEG. Masquage par masques de segmentation (dataset) ou formes, traits et carrés aléatoires, parfois inversés ;
  - extractor : proche de SAM, encodeur patch ViT-base + décodeur pixel ; sortie (1 + nbits) × H × W. Partir de DINO/SAM pré-entraîné n'aide pas beaucoup ;
  - pertes : BCE de détection sur tous les pixels + BCE de décodage seulement sur les pixels marqués (masque vrai) ;
  - fine-tuning multi-messages : k messages mélangés via des masques COCO ;
  - **chiffres (COCO, 10k images de validation)** :
    - au splicing, **HiDDeN, DWTDCT et SSL détectent < 1 %**, TrustMark ~83 % ;
    - **WAM : 97,9 % en inpainting, 100 % en splicing, bit accuracy 95,3 %** ;
    - IoU de localisation ≈ 1 même après un crop à 25 % ; **EditGuard tombe à 0 dès qu'on croppe** (bit accuracy 0,5) ;
    - après inpainting LaMa, 32/32 bits récupérés sur la zone restante.

## 3. Watermarking à la génération
- **Attaques sur l'extractor** :
  - **white-box** (on a l'extractor) : le watermark s'enlève facilement → **l'extractor est gardé privé** ;
  - **black-box** : plus dur mais possible, ex. **DiffPure** (bruit gaussien + débruitage par diffusion). Très dur d'y être robuste : trop coûteux à mettre dans l'entraînement, et même là difficile.
- Post-hoc = régime à haut PSNR, peu de pixels changés → fragile face aux régénérations. D'où le **rêve du watermark sémantique** : bas PSNR, plus dur à enlever.
- **Stable Signature** : fine-tuner le décodeur latent du LDM pour que chaque image générée soit marquée.
- **DistSeal** : embedder dans l'espace latent du générateur.
- **Tree-Ring** : clé dans le bruit initial ; détection par inversion de la diffusion. L'image est complètement différente de la version non marquée.
- **Autorégressif** : réutiliser les techniques de watermark de LLM sur les tokens image.
  - Papiers cités : Jovanovic NeurIPS'25, Tong, Hui, Müller 2026, UniMark 2026, ClusterMark CVPR'26.
  - Tableau (Jovanovic) : sans attaque ou sous transformations valeur/géométrie, le post-hoc est ≈ 1. **Sous DiffPure et compression neuronale, le post-hoc s'effondre ≈ 0**, alors que le watermark de sampling garde ≈ 0,8.

## 4. Problèmes de déploiement (discussion)
- **En production** : DeepMind (SynthID, synthid.com), OpenAI (outil de vérification d'image, mai 2026), Meta, **Anthropic** (texte de Claude, conformité UE, août).
  - **Pour l'image, c'est presque toujours du post-hoc** qui est déployé.
- **Content Seal** est en production sur meta.ai : chaque image générée est marquée. Il a survécu à une impression sur t-shirt puis à une photo. Vérification sur meta.ai/identification.
- **Compromis revisités** : au-delà des 3 critères, il y a l'**adaptabilité, la vitesse et la compacité**, rarement traitées dans les papiers.
- **Freins organisationnels** :
  - l'entreprise doit pouvoir changer de générateur ;
  - fine-tuner le générateur ajoute de la complexité ;
  - les équipes génératives n'aiment pas qu'on touche au sampling ;
  - → **le post-hoc est la méthode de l'industrie**.
- **Robustesse ≠ sécurité** (« Watermarking Security: Theory and Practice », 2005) :
  - robustesse = survivre aux traitements classiques ; sécurité = des non-autorisés ne peuvent pas lire, écrire ou enlever le message ;
  - **99,9 % des modifications sont non intentionnelles** : la robustesse prime ;
  - insistance sur la **robustesse géométrique** (crop, rotation, perspective), point faible de beaucoup de méthodes à la génération.
- Slides sans notes : **rétrocompatibilité** (nouveau modèle vs anciennes images) et **cross-compatibilité de la détection**.
- **Détecteurs publics** : watermarking asymétrique.
  - Exposer l'extractor via une API oui/non : comment la protéger (rate limiting, authentification, détecteur public ≠ privé) ?
  - Extractor open source : on ne sait pas encore se protéger des attaques white-box. **Problème de recherche ouvert jugé critique.**
- **Conclusions** :
  - le watermarking est de plus en plus important (régulateurs, grandes entreprises) ;
  - il améliore beaucoup la détection, toutes modalités confondues ;
  - **génération = plus sûr contre l'enlèvement adverse ; post-hoc = beaucoup plus flexible** ;
  - beaucoup de problèmes pratiques restent ouverts.

## Ce que ça suggère pour le hackathon (inférence)
- Les sujets que l'organisateur met en avant :
  - (a) **contrôle du FPR / tests statistiques et ROC** ;
  - (b) **localisation et splicing** (WAM) ;
  - (c) **attaques black-box type régénération (DiffPure)** et white-box sur l'extractor ;
  - (d) **robustesse géométrique** ;
  - (e) **détecteur public sécurisé** (API, rate limiting, forge/suppression).
- Le format de soumission du cluster (`id,label` avec une probabilité dans [0,1]) colle à une tâche « cette image est-elle marquée ? » évaluée en TPR à FPR bas ou en AUC → calibrer avec la p-value binomiale.
- Liens des papiers du deck : tous téléchargés dans `papers/pdf/` (HiDDeN, TrustMark, SynthID-Image, Video Seal, Pixel Seal, WAM, EditGuard, Stable Signature, DistSeal, Tree-Ring).
