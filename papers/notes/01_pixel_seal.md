# Analyse technique : Pixel Seal (arXiv 2512.16874)

Source : `papers/txt/pixel_seal.txt` (texte intégral, v1 du 18 déc. 2025). Auteurs : Tomáš Souček, Pierre Fernandez, Hady Elsahar, Sylvestre-Alvise Rebuffi, Valeriu Lacatusu, Tuan Tran, Tom Sander, Alexandre Mourachko (Meta FAIR).
Code annoncé dans le papier : https://github.com/facebookresearch/videoseal (site : https://facebookresearch.github.io/meta-seal).

Convention : les faits viennent du papier (numéros d'équations, tableaux et sections cités). Les éléments marqués **(inférence)** sont mes déductions ou interprétations, pas des affirmations du papier.

---

## TL;DR

1. Pixel Seal est un modèle de watermarking invisible multi-bits (256 bits) pour images et vidéos post-hoc, entraîné **uniquement** avec une perte adversariale (discriminateur patch-based) et une perte de message (BCE), sans aucune perte perceptuelle (MSE, LPIPS).
2. Trois mécanismes le rendent stable et robuste : un schedule d'entraînement en 3 étapes (robustesse d'abord, invisibilité ensuite), une atténuation JND calculée à la résolution originale, et une simulation de l'inférence haute résolution pendant l'entraînement. La vidéo passe par un "temporal watermark pooling" sans réentraînement.
3. Résultats : sur l'attaque combinée (crop + JPEG + luminosité) des images Meta AI, Pixel Seal atteint 0,91 de bit accuracy sur 256 bits (−log10 p = 50,3), contre 0,94 pour InvisMark mais sur 94 bits seulement (−log10 p = 22,1). Imperceptibilité parmi les meilleures (LPIPS, CVVDP, JND), sauf PSNR/SSIM où Video Seal 0.0 domine grâce à des artefacts très localisés mais visibles.

---

## Problème

Le papier identifie trois verrous des méthodes existantes (section 1 et 4) :

1. **Proxys perceptuels imparfaits.** Les pertes MSE ou LPIPS ne distinguent pas les zones lisses des zones texturées, alors que l'humain remarque mieux les artefacts dans les zones lisses (ciel, aplats). Le papier cite aussi les modèles de Watson (Czolbe et al., 2020) et la focal frequency loss comme alternatives, toutes jugées imparfaites.
2. **Objectifs contradictoires.** L'optimum de la perte perceptuelle est un watermark nul (rien n'est embarqué). L'optimum de la robustesse donne un watermark très visible. Sans réglage précis, l'entraînement échoue : soit le modèle n'embarque rien, soit il cache si bien que le décodeur ne retrouve plus le message. Le papier dit que les multiples initialisations aléatoires donnent des résultats très différents.
3. **Passage en haute résolution.** Les modèles sont entraînés sur des crops 256×256. À l'inférence, l'image est réduite puis le watermark réagrandi par interpolation. Cela crée des artefacts visibles autour des contours (le watermark "déborde" dans les zones plates), des artefacts invisibles en basse résolution deviennent visibles en haute résolution, et l'extracteur voit une distribution différente à l'inférence (out-of-distribution, Éq. 9-10).

Le papier vise aussi la robustesse aux recadrages, filtres type Instagram et compression, sans compromettre l'imperceptibilité.

---

## Méthode en détail

### 1. Cadre de base (section 3)

- **Embedder** : prend `x ∈ [0,1]^{H×W×3}` et un message binaire `m ∈ {0,1}^{nbits}`, produit `w ∈ [-1,1]^{H×W×3}`. Image watermarkée : `xw = x + α·w` (Éq. 1).
- **Extracteur** : prend une image possiblement altérée et sort un message "mou" `m̃ ∈ [0,1]^{nbits}`, seuillé en `m̂ ∈ {0,1}^{nbits}`.
- **Détection avec garantie statistique** (Éq. 2) : sous l'hypothèse qu'une image non watermarkée donne des messages équiprobables, la probabilité qu'un message aléatoire soit à distance de Hamming ≤ d est `p = Σ_{k≤d} C(n,k) / 2^n`. Il n'y a pas besoin de calibrer un taux de faux positifs sur des images non watermarkées, contrairement aux méthodes zero-bit. C'est l'argument principal pour le multi-bits.
- **Métrique de robustesse** : `−log10 p` est utilisé pour comparer des méthodes avec des longueurs de message différentes (renvoi à Fernandez et al. 2024, annexe A).

### 2. Architecture (section 5.1)

- **Embedder** : U-Net (Saharia et al., 2022), **43,8 M de paramètres**.
- **Extracteur** : ConvNeXt-v2 Tiny (Woo et al., 2023), **33,4 M de paramètres**.
- **Message** : `nbits = 256`.
- **Discriminateur** : patch-based, celui de Rombach et al. (2022, latent diffusion). Le papier dit qu'il offre "un bon équilibre" : un discriminateur plus efficace enlève mieux les artefacts visibles, mais rend le watermark moins robuste.

### 3. Pertes (sections 3.2 et 4.1)

**Forme générale** (Éq. 4) :
`L = λperc·Lperc + λadv·Ladv + λmsg·Lmsg`

**Perte de message** `Lmsg` (Éq. 3) : entropie croisée binaire moyennée sur les bits, appliquée à la sortie de l'extracteur `m̃`, rétropropagée à travers l'extracteur et l'embedder.

**Perte adversariale** (Éq. 6), version "adversarial-only" de Pixel Seal :
- Mise à jour du discriminateur : `½ ∇θD [ ReLU(1 − D(x)) + ReLU(1 + D(x̂w)) ]` (hinge loss à double terme, Lim & Ye 2017, GeometricGAN).
- Mise à jour de l'embedder : `− λadv ∇θEmb D(x̂w)`.
- Ici `x̂w` est l'image watermarkée **boostée** (voir ci-dessous), et `x` l'image originale.

**Perte perceptuelle** : **absente** dans le modèle publié (`λperc = 0`). Le papier justifie ce choix par l'ablation (tableau 3a) : ajouter une MSE avec `λperc = 0,1` ne change presque rien, et avec `λperc = 1,0` le modèle n'apprend plus rien (bit accuracy 50 %).

**Valeurs retenues dans le modèle publié** : `λmsg = 1,0`, `λadv = 0,1`, `λperc = 0`.

### 4. Boosting du watermark (section 4.1)

`x̂w = x + β·(xw − x)` remplace `xw` dans la perte adversariale (Éq. 6).
- `β > 1` : le watermark est amplifié pour le discriminateur, qui le détecte et l'enlève plus facilement. Résultat : watermark plus imperceptible, mais moins robuste.
- `β < 1` : watermark plus robuste mais plus visible.
- Modèle publié : `β = 1`. Figure 1 et tableau 3b montrent trois modèles (β = 0,5 ; 1 ; 2,5).

### 5. Augmentations et gradient (section 3.2)

- Pendant l'entraînement, `xw` passe par des transformations aléatoires avant l'extracteur : géométriques (crop, resize, rotation), valuemétriques (luminosité, contraste), compression (JPEG).
- Pour les transformations non différentiables (JPEG notamment), le papier utilise un **straight-through estimator** (Éq. 5) : `x̃w = xw + nograd(T(xw) − xw)`, soit gradient identité. Le papier cite aussi les approches à base de réseau approximant la transformation (Luo et al. 2023 ; Shen et al. 2023).
- Les paramètres exacts et probabilités d'augmentation à l'entraînement ne sont pas détaillés dans le texte principal **(inférence : ils sont probablement dans le code ou en annexe)**.

### 6. Atténuation JND (section 4.1, Éq. 7)

- `xw = x + α·(w ⊙ JND(x))`.
- `JND : [0,1]^{H×W×3} → [0,1]^{H×W}` : **fonction non apprenable**, qui donne de grandes valeurs là où une modification de l'intensité passe inaperçue (contours, texture) et de petites valeurs ailleurs. Le papier utilise la carte JND de Zhang et al. (2008), comme Sander et al. (2025).
- Effet : le watermark est confiné aux zones à forte variance de pixels, et ne va pas dans les aplats.
- Tableau 3a : sans JND, PSNR 43,1 contre 47,0 avec JND, et JND-metric 1,34 contre 0,44.

### 7. Schedule en trois étapes (section 4.2)

Motivation : les optima des pertes perceptuelles et adversariales correspondent à `w = 0`, donc l'entraînement est fragile.

- **Étape 1** : seule `Lmsg` est utilisée, avec `α = α0 = 1,0` (watermark très visible mais facile à apprendre). On entraîne jusqu'à saturation de la bit accuracy.
- **Étape 2** : on ajoute `Ladv` et on fait décroître `α` de `α0` vers `α1 = 0,2` selon un cosinus (Éq. 8) :
  - `α(t) = α1 + (α0 − α1)·cos((π/2)·φ)`
  - `φ = clip((t − Nstart) / (Nend − Nstart), 0, 1)`
  - **Les valeurs de Nstart et Nend ne sont pas données dans le texte principal (inférence : à chercher dans le code ou l'annexe).**
- **Étape 3** : fine-tuning avec `α = α1`, éventuellement sur d'autres données ou attaques, pour une application spécifique.

Ablations (tableau 3c) : sans étape 1 (α fixé à 0,2 dès le début) ou sans délai du discriminateur, le modèle reste à 50 % de bit accuracy.

### 8. Haute résolution (section 4.3)

**Baseline (Éq. 9)** : `xw = x + α·resize↑(w)`, `w = Emb(resize↓(x), m)`. Problèmes listés par le papier : débordement des artefacts, artefacts visibles en haute résolution, distribution décalée pour l'extracteur.

**Pipeline Pixel Seal (Éq. 10)** :
- `m̃ = Ext(x̃w)`, `x̃w = resize↓(τ(xw))`
- `xw = x + α·(resize↑(w) ⊙ JND(x))`
- `w = Emb(resize↓(x), m)`

Concrètement :
- **JND calculé à la résolution originale** : moins de débordement dans les zones plates. Le coût d'inférence augmente peu par rapport à l'embedder.
- **Perte adversariale calculée à la résolution originale** : élimine les artefacts visibles seulement en haute résolution. Coût d'entraînement augmenté, sans effet à l'inférence.
- **Simulation de l'inférence pendant l'entraînement** : l'extracteur voit `resize↓(τ(xw))`, donc la même distribution qu'à l'inférence. Sans effet à l'inférence.
- `τ` est une augmentation aléatoire **différentiable**. Les attaques non différentiables ne sont pas dans la boucle d'entraînement de cette version.

Paramètres d'entraînement haute résolution (section 5.1) :
- Taille modèle : 256×256 pour l'embedder.
- Taille maximale d'entraînement `Smax = 768`, pour des raisons pratiques.
- Redimensionnement aléatoire de largeur/hauteur dans `[Smin, Smax] = [256, 768]`, pour la robustesse au ratio d'aspect.

### 9. Vidéo : temporal watermark pooling (section 4.4)

Les deux difficultés citées : l'entraînement vidéo est coûteux (décodage, pré-traitement, faible diversité dans un batch), et le watermarking frame par frame est trop lent.

Alternatives écartées par le papier :
- Watermarker une frame sur k : papillotement si le watermark n'est pas parfaitement invisible, vulnérabilité à la compression, scan exhaustif à l'extraction.
- Propager le watermark de la frame k aux k−1 autres (Video Seal 0.0 et 1.0) : artefacts de "ghosting", nécessite un finetuning vidéo avec le k exact.

**Méthode Pixel Seal** : inférence uniquement, aucun entraînement vidéo.
- Insertion d'une **moyenne temporelle** (avg pooling, kernel = stride = k) dans l'embedder, après le d-ième bloc de downsampling du U-Net.
- Insertion d'un **un-pooling temporel** (répétition) au même niveau dans les blocs de upsampling.
- Justification : les frames voisines ont des features sémantiques haut niveau très proches, donc on les moyenne ; les détails bas niveau par frame sont calculés par les dernières couches, qui ne sont pas poolées.
- Paramètres utilisés : `k = 4` (section 5.2), et profondeur `n = 2` dans le texte de la section 5.2, alors que le tableau 4 utilise `d`. **Incohérence de notation entre le texte et le tableau, à garder en tête.**

Tableau 4 (SA-V, 96 vidéos) :

| Configuration | Bit acc. combinée | PSNR | Speedup |
|---|---|---|---|
| k = 1 (baseline) | 0,68 | 47,6 | — |
| k = 4, d = 3 | 0,69 | 47,7 | 1,0× |
| k = 4, d = 2 | 0,69 | 47,6 | 1,7× |
| k = 4, d = 1 | 0,66 | 47,7 | 2,2× |

Vitesse mesurée sur Quadro GP100, 64 frames, PyTorch vanilla.

### 10. Hyperparamètres clés (récapitulatif)

| Paramètre | Valeur | Source |
|---|---|---|
| Message | 256 bits | section 5.1 |
| Embedder | U-Net, 43,8 M params | section 5.1 |
| Extracteur | ConvNeXt-v2 Tiny, 33,4 M params | section 5.1 |
| Données d'entraînement | SA-1B | section 5.1 |
| Steps | 600 000 | section 5.1 |
| Batch size | 256 | section 5.1 |
| Optimiseur | AdamW | section 5.1 |
| Learning rate | 5×10⁻⁴, cosinus | section 5.1 |
| Warm-up | 20 000 steps | section 5.1 |
| Résolution d'entraînement | aléatoire dans [256, 768] | section 5.1 |
| α0 / α1 | 1,0 / 0,2 | section 5.1 |
| β (modèle publié) | 1 | section 5.1 |
| λmsg / λadv / λperc | 1,0 / 0,1 / 0 | section 5.1 |
| Pooling vidéo | k = 4 | section 5.2 |

Valeurs non précisées dans le texte principal : Nstart, Nend, nombre de steps par étape, probabilités et amplitudes des augmentations d'entraînement, valeur exacte du seuil de décision des bits.

---

## Setup expérimental

### Datasets

**Images** :
- **1000 images Meta AI, 1280×1280**, générées à partir des prompts de Gowal et al. (2025, SynthID-Image).
- **100 photos réelles haute résolution du set de validation SA-1B** (Kirillov et al., 2023).

**Vidéos** :
- **121 vidéos Movie Gen** (Polyak et al., 2024). Le lien YouTube donné dans le papier pointe vers une playlist.
- **96 vidéos du set de validation SA-V**.
- Évaluation sur les **3 premières secondes** de chaque vidéo.

**Entraînement** : SA-1B (images), uniquement.

**Inférence / évaluation sur les modèles ablations** : 100 photos SA-1B validation (tableau 3). Table 4 : 96 vidéos SA-V.

### Métriques

**Imperceptibilité** :
- **PSNR** et **SSIM** (classiques).
- **LPIPS** (Zhang et al., 2018).
- **CVVDP** (ColorVideoVDP, Mantiuk et al., 2024).
- **JND** : métrique maison, entraînée sur des données de user study internes. Une différence de 1 JND signifie que 75 % des observateurs la remarquent. **Le papier précise que cette métrique n'est pas la carte JND utilisée pour l'atténuation.** Elle est non publiée (données internes).

**Robustesse** :
- **Bit accuracy** (sur les 256 bits pour Pixel Seal).
- **−log10 p** : log de la probabilité d'observer la bit accuracy mesurée par hasard (Éq. 2). Permet de comparer des méthodes avec des nombres de bits différents.

### Attaques et augmentations testées

Liste complète dans la figure 8 (images et vidéos) et visualisations en figure 9. Le papier ne donne pas de tableau séparé, donc ce qui suit vient de la figure 8.

**Images** :
- **Identité**.
- **Valuemétriques** : luminosité 0,1 à 2,0 ; contraste 0,1 à 2,0 ; teinte −0,4 à 0,5 ; niveaux de gris ; flou gaussien 3, 5, 9, 13, 17.
- **Géométriques** : flip horizontal ; rotation 5°, 10°, 30°, 45°, 90° ; crop (de 0,32 à 1,0 de la surface ou du côté, valeur exacte à confirmer dans l'annexe) ; perspective 0,1 à 0,8.
- **Compression** : JPEG qualité 40, 50, 60, 70, 80, 90.
- **Combinées** : (JPEG, crop, luminosité), par exemple (40, 0,71, 0,5), (60, 0,71, 0,5), (80, 0,71, 0,5).

**Vidéos** :
- **Valuemétriques** : luminosité 0,5 et 1,5 ; contraste 0,5 et 1,5 ; saturation 0,5 et 1,5 ; teinte 0,25 ; niveaux de gris ; flou gaussien 9.
- **Géométriques** : flip ; rotation 10° et 90° ; crop 0,55 et 0,71 ; perspective 0,5.
- **Compression** : JPEG 40 ; H.264 CRF 23, 30, 40, 50 ; H.264rgb CRF 23, 30, 40, 50 ; H.265 CRF 23, 30, 40, 50.
- **Combinées** : (H.264, crop, luminosité) avec CRF 23, 30, 40, 50.

**Note** : figure 1 décrit l'attaque combinée comme "luminosité 0,5, crop 50 % de surface, JPEG qualité 40". La figure 8 utilise `crop 0,71`. **(inférence) : 0,71 ≈ √0,5, donc le crop 0,71 correspondrait à 50 % de surface si le paramètre est le côté.**

**Attaques non évaluées dans le papier** (inférence, à partir de la liste ci-dessus) :
- Régénération par modèle de diffusion (purification), alors que le papier cite Pan et al. (2024) et Lu et al. (2025) comme travaux sur ce sujet.
- Attaques adversariales ou à base d'apprentissage pour effacer le watermark.
- Falsification : copie d'un watermark sur une autre image, ou extraction puis réinjection.
- Attaques en boîte noire ou avec connaissance du modèle.

---

## Résultats chiffrés clés

Les tableaux ci-dessous reprennent les valeurs du papier. Format : bit accuracy / −log10 p (sauf mention).

### Tableau 1 : robustesse, images Meta AI (1000 images)

| Méthode | Identité | Valuemétrique | Compression | Géométrique | Combinée |
|---|---|---|---|---|---|
| CIN (30 bits) | 1,00 / 9,0 | 0,85 / 7,5 | 1,00 / 9,0 | 0,52 / 0,7 | 0,50 / 0,4 |
| InvisMark (94 bits) | 1,00 / 28,3 | 0,89 / 22,0 | 0,98 / 26,1 | 0,77 / 14,8 | **0,94 / 22,1** |
| MBRS (256 bits) | 1,00 / 75,9 | 0,95 / 62,0 | 1,00 / 75,2 | 0,52 / 3,6 | 0,50 / 0,4 |
| TrustMark-P | 1,00 / 29,6 | 0,89 / 21,2 | 0,98 / 27,1 | 0,65 / 8,2 | 0,52 / 0,7 |
| TrustMark-Q | 1,00 / 30,0 | 0,97 / 27,1 | 1,00 / 29,8 | 0,67 / 9,4 | 0,54 / 0,9 |
| Video Seal 0.0 (96 bits) | 0,97 / 25,0 | 0,84 / 16,7 | 0,93 / 20,9 | 0,80 / 13,3 | 0,74 / 7,4 |
| Video Seal 1.0 (256 bits) | 0,98 / 69,4 | 0,95 / 60,7 | 0,95 / 60,6 | 0,83 / 42,9 | 0,69 / 16,0 |
| WAM (32 bits) | 1,00 / 9,6 | 0,89 / 7,5 | 1,00 / 9,5 | 0,77 / 4,7 | 0,62 / 1,6 |
| **Pixel Seal** | **1,00 / 75,8** | **0,97 / 68,1** | **0,98 / 68,9** | **0,95 / 63,9** | **0,91 / 50,3** |

### Tableau 1 : robustesse, photos SA-1B (100 photos)

| Méthode | Identité | Valuemétrique | Compression | Géométrique | Combinée |
|---|---|---|---|---|---|
| InvisMark | 1,00 / 28,3 | 0,89 / 21,4 | 1,00 / 27,8 | 0,77 / 14,1 | **0,98 / 25,7** |
| MBRS | 0,99 / 71,8 | 0,93 / 55,9 | 0,99 / 71,1 | 0,52 / 3,4 | 0,50 / 0,4 |
| TrustMark-P | 0,99 / 28,8 | 0,84 / 17,3 | 0,98 / 26,7 | 0,64 / 7,7 | 0,52 / 0,6 |
| TrustMark-Q | 1,00 / 29,9 | 0,96 / 25,9 | 1,00 / 29,7 | 0,65 / 8,5 | 0,53 / 0,8 |
| Video Seal 0.0 | 0,98 / 25,2 | 0,85 / 17,2 | 0,96 / 23,3 | 0,79 / 12,6 | 0,74 / 7,6 |
| Video Seal 1.0 | 0,99 / 72,8 | 0,96 / 64,6 | 0,99 / 71,0 | 0,82 / 41,8 | 0,70 / 17,1 |
| WAM | 1,00 / 9,6 | 0,90 / 7,7 | 1,00 / 9,6 | 0,78 / 4,8 | 0,72 / 3,0 |
| **Pixel Seal** | **1,00 / 75,4** | **0,98 / 68,5** | **0,99 / 73,6** | **0,93 / 59,0** | **0,94 / 55,6** |

**Point d'attention** : sur la combinée SA-1B, InvisMark a une bit accuracy plus haute (0,98 contre 0,94) mais un −log10 p plus bas (25,7 contre 55,6). La bit accuracy seule n'est pas comparable entre méthodes à nombre de bits différent, d'où l'usage de −log10 p. Le papier écrit que "seuls Pixel Seal et InvisMark restent au-dessus de 90 %", ce qui est exact.

### Tableau 1 : robustesse, vidéos

**MovieGen (121 vidéos)** :

| Méthode | Identité | Valuemétrique | Compression | Géométrique | Combinée |
|---|---|---|---|---|---|
| RivaGAN (32 bits) | 0,93 / 7,2 | 0,83 / 5,1 | 0,81 / 4,5 | 0,58 / 1,0 | 0,56 / 0,7 |
| Video Seal 0.0 | 0,98 / 26,3 | 0,88 / 20,0 | 0,86 / 16,7 | 0,82 / 14,4 | 0,70 / 6,9 |
| Video Seal 1.0 | 1,00 / 76,1 | 1,00 / 74,9 | **0,89 / 51,5** | 0,88 / 50,0 | 0,63 / 12,1 |
| **Pixel Seal** | **1,00 / 76,9** | **1,00 / 76,5** | 0,86 / 48,6 | **0,97 / 68,0** | **0,70 / 25,6** |

**SA-V (96 vidéos)** :

| Méthode | Identité | Valuemétrique | Compression | Géométrique | Combinée |
|---|---|---|---|---|---|
| RivaGAN | 0,97 / 8,2 | 0,87 / 6,1 | 0,82 / 4,7 | 0,60 / 1,3 | 0,56 / 0,7 |
| Video Seal 0.0 | 0,99 / 27,3 | 0,89 / 21,1 | 0,81 / 14,6 | 0,83 / 15,7 | 0,64 / 4,5 |
| Video Seal 1.0 | 1,00 / 76,6 | 1,00 / 76,0 | **0,87 / 49,4** | 0,87 / 50,4 | 0,59 / 8,0 |
| **Pixel Seal** | **1,00 / 77,1** | **1,00 / 76,8** | 0,86 / 48,1 | **0,98 / 70,8** | **0,68 / 24,2** |

**Point d'attention** : en compression vidéo (JPEG 40 et H.264/H.265 non combinés), Video Seal 1.0 fait légèrement mieux que Pixel Seal sur MovieGen (51,5 contre 48,6) et SA-V (49,4 contre 48,1). Le texte dit pourtant que Pixel Seal "surpasse toutes les méthodes liées" (section 5.2), ce qui n'est pas strictement vrai pour cette colonne.

### Tableau 2 : imperceptibilité, 1000 images Meta AI

| Méthode | PSNR ↑ | SSIM ↑ | CVVDP ↑ | LPIPS ↓ | JND ↓ |
|---|---|---|---|---|---|
| CIN | 44,3 | 0,9894 | 9,51 | 0,0274 | 1,93 |
| InvisMark | 46,6 | 0,9823 | 9,93 | 0,0020 | 0,26 |
| MBRS | 46,3 | 0,9928 | 9,54 | 0,0049 | 1,59 |
| TrustMark-P | 49,0 | 0,9908 | **9,94** | 0,0018 | **0,19** |
| TrustMark-Q | 43,3 | 0,9886 | 9,73 | 0,0022 | 1,66 |
| Video Seal 0.0 | **53,7** | **0,9989** | 9,85 | 0,0039 | 1,07 |
| Video Seal 1.0 | 48,7 | 0,9958 | 9,84 | **0,0012** | 0,45 |
| WAM | 41,4 | 0,9760 | 9,72 | 0,0344 | 0,95 |
| **Pixel Seal** | 48,9 | 0,9905 | 9,87 | 0,0013 | 0,34 |

**Vérification du claim "top 3 sur 4 métriques sur 5"** (inférence, car calcul fait par moi) : Pixel Seal est 3e en PSNR, 3e en CVVDP, 2e en LPIPS (derrière Video Seal 1.0) et 3e en JND (derrière TrustMark-P et InvisMark). Il est 4e en SSIM (0,9905, derrière MBRS 0,9928 et TrustMark-P 0,9908). Le claim est donc vrai.

### Tableau 3 : ablations (100 photos SA-1B)

| Config | Identité | Valuemétrique | Compression | Géométrique | Combinée | PSNR | JND |
|---|---|---|---|---|---|---|---|
| **Pixel Seal (référence)** | 1,00 | 0,98 | 0,99 | 0,93 | 0,94 | 47,0 | 0,44 |
| + perte perceptuelle λ=0,1 | 0,99 | 0,97 | 0,99 | 0,92 | 0,91 | 47,1 | 0,41 |
| + perte perceptuelle λ=1,0 | 0,50 | 0,50 | 0,50 | 0,50 | 0,50 | — | — |
| sans discriminateur (λadv=0) | 1,00 | 0,99 | 1,00 | 0,95 | 0,97 | 44,9 | 0,96 |
| sans JND | 1,00 | 0,98 | 1,00 | 0,93 | 0,97 | 43,1 | 1,34 |
| β = 0,5 | 1,00 | 0,98 | 1,00 | 0,95 | 0,96 | 46,2 | 0,56 |
| β = 2,5 | 0,99 | 0,96 | 0,98 | 0,90 | 0,86 | 50,1 | 0,30 |
| sans mise à l'échelle (α0 = α1 = 0,2) | 0,50 | 0,50 | 0,50 | 0,50 | 0,50 | — | — |
| sans délai du discriminateur | 0,50 | 0,50 | 0,50 | 0,50 | 0,50 | — | — |
| résolution fixe 256×256 | 0,97 | 0,94 | 0,96 | 0,81 | 0,68 | 46,8 | 0,45 |

Lecture : le discriminateur et le JND achètent l'imperceptibilité (JND 0,44 contre 0,96 et 1,34) au prix d'un peu de robustesse combinée (0,94 contre 0,97). Le boost β règle ce compromis. La haute résolution gagne fortement en combinée (0,94 contre 0,68).

### Tableau 4 : pooling temporel

Voir section "Vidéo" ci-dessus. Résumé : k = 4, d = 1 donne 2,2× de vitesse, pour une baisse de bit accuracy combinée de 0,68 à 0,66 (k = 1 contre k = 4).

---

## Limites

Le papier n'a pas de section "Limitations" dédiée. Les points suivants viennent du texte ou de constats directs sur les tableaux.

**Limites reconnues ou explicites dans le papier :**
- **Sensibilité à l'initialisation et aux hyperparamètres** des méthodes existantes, que Pixel Seal atténue mais dont il ne supprime pas la contrainte (le schedule en 3 étapes est une solution empirique "observée").
- **Compromis robustesse / imperceptibilité** : le boost β et le discriminateur déplacent le point de fonctionnement, sans le lever.
- **Intérêt limité du discriminateur** : un discriminateur plus puissant enlève mieux les artefacts mais réduit la robustesse.
- **Métrique JND non publique** : elle repose sur des données de user study internes, donc non reproductible hors Meta.
- **Certaines attaques restent difficiles** : sur la combinée (H.264, crop, luminosité) en vidéo, Pixel Seal tombe à 0,70 (MovieGen) et 0,68 (SA-V) de bit accuracy.
- **Le papier signale que la compression vidéo (H.264, H.265) pèse plus que la compression image** : "significantly greater challenge".
- **Vidéo évaluée sur 3 secondes seulement**.
- **Le texte ne couvre pas la vitesse absolue sur GPU moderne** : la mesure est faite sur un Quadro GP100 (GPU ancien).

**Limites que je déduis (inférence) :**
- **Entraînement limité à 768 px** : la résolution de 1280 px des tests est atteinte par inférence, pas par entraînement direct. L'effet exact sur les très hautes résolutions (4K, 8K) n'est pas testé.
- **Pas d'attaque par régénération (diffusion) testée** : or c'est l'attaque la plus réaliste contre les watermarks IA.
- **Pas de test de falsification ni de copie de watermark**.
- **Données d'évaluation majoritairement synthétiques** (1000 images Meta AI contre 100 photos SA-1B pour les images), et les images Meta AI viennent du même générateur que les prompts SynthID.
- **Message de 256 bits** : il donne une garantie statistique forte (−log10 p), mais la charge utile fixe est une contrainte pour les applications qui en veulent moins ou plus.
- **Modèle publié avec β = 1 et λadv = 0,1**, mais les tableaux montrent des configurations plus robustes ou plus imperceptibles. L'utilisateur doit choisir le bon checkpoint.
- **Incohérences de notation** : "n = 2" dans le texte contre "d" dans le tableau 4 pour la profondeur du pooling ; crop "50 % de surface" dans la figure 1 contre "0,71" dans la figure 8.

---

## Code et ressources

- **Papier** : arXiv:2512.16874 (v1, 18 déc. 2025).
- **Code** : https://github.com/facebookresearch/videoseal (annoncé par le papier, non vérifié ici car je n'ai pas accès au web dans cette session).
- **Site** : https://facebookresearch.github.io/meta-seal
- **Poids** : le papier dit "We release Pixel Seal model weights", sans lien direct. Ils sont probablement dans le dépôt videoseal **(inférence)**.
- **Modèles de référence à inclure dans les comparaisons** : Video Seal 1.0 (Fernandez et al., 2024), WAM (Sander et al., 2025), TrustMark (Bui et al., 2023), InvisMark (Xu et al., 2025), MBRS (Jia et al., 2021), CIN (Ma et al., 2022), RivaGAN (Zhang et al., 2019).
- **Outils d'évaluation mentionnés** : LPIPS, ColorVideoVDP (CVVDP).
- **Données** : SA-1B (Kirillov et al., 2023), SA-V, Movie Gen (Polyak et al., 2024) pour les vidéos (liens de playlist dans le papier).

---

## Idées de challenges hackathon

### Challenge A : attaque (retirer ou dégrader le watermark)

**Objectif** : faire chuter la bit accuracy de Pixel Seal sous 0,5 (aléatoire) tout en gardant l'image visuellement proche de l'original.

**Pistes concrètes** :
1. **Purification par diffusion** : passer l'image watermarkée dans un modèle de diffusion (img2img à faible force de bruit, puis débruitage). Le papier ne teste pas ce cas. Mesurer la bit accuracy après régénération, comparer à la valeur d'origine (0,91 en combinée Meta AI).
2. **Attaque par optimisation** : minimiser la bit accuracy par descente de gradient sur l'image, sous contrainte de distance (LPIPS, PSNR). Pixel Seal est différentiable, donc une attaque white-box est possible.
3. **Chaîne d'attaques réalistes** : recadrage + JPEG + recompression H.264 + légère correction de couleur. Le papier montre que la compression vidéo combinée est la plus dure (0,68 à 0,70).
4. **Attaque par transfert** : entraîner un petit extracteur de substitution, puis optimiser une perturbation contre lui, et tester sur Pixel Seal.

**Métriques** : bit accuracy, −log10 p (Éq. 2) pour rester comparable aux tableaux du papier, PSNR, LPIPS.

### Challenge B : défense (rendre le watermark plus robuste)

**Objectif** : améliorer la bit accuracy sous les attaques du papier, en particulier la combinée et la géométrique (0,91 et 0,95 respectivement).

**Pistes concrètes** :
1. **Ajouter des attaques d'entraînement** : purification par diffusion ou un réseau approximant H.264 (le papier cite Luo et al. 2023 et Shen et al. 2023 comme approches à réseau de substitution).
2. **Réduire la dépendance au discriminateur** : tester un discriminateur plus léger ou un schedule différent, et mesurer le compromis via le tableau 3a.
3. **Ensemble de messages** : répéter le message sur plusieurs canaux ou régions, puis voter. (Inférence : pas testé dans le papier.)
4. **Pooling vidéo adaptatif** : un k dépendant de la complexité temporelle, pour garder la robustesse sans perdre de vitesse.

**Protocole** : fixer les 1000 images Meta AI ou 100 photos SA-1B. Reprendre les attaques de la figure 8. Rapporter la bit accuracy et le −log10 p.

### Challenge C : détection (savoir si une image est watermarkée)

**Objectif** : décider si une image est watermarkée ou non, avec un taux de faux positifs contrôlé.

**Pistes concrètes** :
1. **Test statistique** : extraire le message, calculer la distance de Hamming au message de référence, et la p-valeur de l'Éq. 2. Fixer un seuil à p < 10⁻⁶ ou similaire. La garantie statistique ne demande pas de calibration sur des images non watermarkées, selon le papier.
2. **Détection sans clé** : entraîner un classifieur sur les sorties de l'extracteur (distribution des logits `m̃`), pour détecter une présence de watermark même si le message est inconnu. (Inférence : le papier ne traite pas ce cas.)
3. **Détection après attaque** : construire une courbe ROC bit accuracy / −log10 p sur les attaques de la figure 8.

**Protocole** : images watermarkées avec le message de référence contre images réelles SA-1B et images Meta AI non watermarkées. Mesurer la courbe ROC.

### Pistes d'implémentation pour un hackathon

**Étape 1 : cloner et charger**
- Cloner le dépôt videoseal (lien ci-dessus).
- Charger les poids Pixel Seal (suivre le README du dépôt, non vérifié ici).
- Faire un premier test : une image 1280×1280 SA-1B, encoder un message aléatoire de 256 bits, décoder, vérifier la bit accuracy sans attaque (doit être ≈ 1,00).

**Étape 2 : banc d'évaluation**
- Coder les attaques de la figure 8 (JPEG via PIL, crop, luminosité, flou, rotation, perspective).
- Implémenter le −log10 p de l'Éq. 2 : `p = Σ_{k≤d} C(256, k) / 2^256`, en log pour éviter le débordement (utiliser `scipy.special.gammaln` ou `math.lgamma`, ou `mpmath`).
- Mesurer PSNR, SSIM, LPIPS (pip `lpips`). La CVVDP et la métrique JND du papier ne sont pas reproductibles simplement.

**Étape 3 : attaque ou défense**
- Pour l'attaque : PyTorch, boucle d'optimisation de la perturbation, perte = −BCE(message) + contrainte de distance.
- Pour la défense : fine-tuning de l'extracteur seul sur les attaques supplémentaires (plus rapide que tout réentraîner), avec un embedder gelé.

**Étape 4 : démo**
- Interface minimale (Gradio ou Streamlit) : upload d'une image, encodage, attaque sélectionnable, décodage et affichage de la bit accuracy et du −log10 p.

**Risques** :
- Le modèle publié est lourd (≈ 77 M de paramètres au total) : prévoir un GPU, sinon l'inférence sur 1280 px sera lente **(inférence)**.
- Les poids ne sont pas garantis compatibles avec une version PyTorch récente **(inférence)**.
- Le fine-tuning complet (600 000 steps, batch 256) n'est pas faisable en hackathon. Limiter à l'extracteur, ou à un checkpoint existant.

---

## Glossaire

- **Watermarking invisible** : insertion d'une information dans un contenu numérique sans dégradation visible.
- **Multi-bits** : watermark qui porte un message binaire long (ici 256 bits), par opposition au zero-bit (présence / absence seulement).
- **Embedder** : réseau qui produit le watermark `w` à partir de l'image et du message.
- **Extracteur** : réseau qui retrouve le message à partir de l'image (possiblement altérée).
- **Bit accuracy** : fraction des bits correctement décodés.
- **−log10 p** : log de la probabilité qu'une bit accuracy observée arrive par hasard. Compare des méthodes à nombres de bits différents.
- **PSNR** : rapport signal sur bruit de crête, en dB. Plus haut = plus proche de l'original.
- **SSIM** : similarité structurelle, entre 0 et 1. Plus haut = mieux.
- **LPIPS** : distance perceptuelle apprise sur des features de réseau profond. Plus bas = mieux.
- **CVVDP** : ColorVideoVDP, prédicteur de différence visuelle pour images et vidéos. Plus haut = mieux.
- **JND (Just Noticeable Difference)** : seuil en dessous duquel une modification passe inaperçue. Ici, deux usages distincts : (1) carte d'atténuation calculée par Zhang et al. (2008), (2) métrique maison du papier, entraînée sur des user studies internes.
- **Perte adversariale (GAN)** : un discriminateur apprend à distinguer images originales et watermarkées ; l'embedder cherche à le tromper.
- **Hinge loss à double terme** : formulation de la perte du discriminateur (Lim & Ye, 2017).
- **Patch-based discriminator** : discriminateur qui juge des patchs locaux plutôt que l'image entière.
- **Boosting (β)** : amplification de la différence `xw − x` pour le discriminateur, qui contrôle le compromis imperceptibilité / robustesse.
- **Straight-through estimator** : approximation du gradient d'une opération non différentiable par l'identité.
- **Simulation de l'inférence** : pendant l'entraînement, reproduire exactement le pipeline de resize utilisé à l'inférence, pour éviter une distribution décalée.
- **Temporal watermark pooling** : moyenne temporelle des features de frames voisines à l'inférence, pour accélérer le traitement vidéo.
- **U-Net** : architecture encodeur-décodeur à connexions de saut, utilisée ici pour l'embedder.
- **ConvNeXt-v2** : architecture convolutive moderne utilisée pour l'extracteur.
- **SA-1B / SA-V** : jeux de données de Meta (Segment Anything), images et vidéos.
- **Movie Gen** : modèle de génération vidéo de Meta, dont le jeu de vidéos sert à l'évaluation.
- **Diffusion purification** : attaque qui repasse l'image dans un modèle de diffusion pour effacer le watermark.
- **Straight-through / white-box / boîte noire** : niveaux d'accès du modèle attaqué (white-box = poids et gradients connus).
