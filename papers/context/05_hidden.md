# CONTEXTE AGENT — HiDDeN: Hiding Data With Deep Networks (Stanford, ECCV 2018, arXiv 1807.09937)
Source vérifiée : texte complet lu (`papers/txt/hidden.txt`). Code original : non fourni par les auteurs. **Poids utilisables : baseline « HiDDeN 48 bits replicate » de Meta dans `repos/videoseal/ckpts/hidden_{encoder,decoder}_48b.pt`** (`build_baseline('hidden')`, `scaling_w = 0.2`).

## En une phrase
Premier watermarking/stéganographie **profond entraîné de bout en bout** : un encodeur cache un message binaire dans l'image, une **couche de bruit** (édits) l'abîme, un décodeur relit le message ; un discriminateur adverse améliore l'invisibilité. C'est « la recette » que reprennent TrustMark, Video Seal, Pixel Seal et WAM (slide organisateur n°15).

## Pipeline & formules
- Composants : encodeur `Eθ`, couche de bruit sans paramètres `N`, décodeur `Dφ`, adversaire `Aγ`.
- `Ien = E(Ico, Min)`, `Min ∈ {0,1}^L` ; `Ino = N(Ico, Ien)` (peut changer la taille : crop) ; `Mout = D(Ino)`.
- Pertes :
  - image : `LI = ‖Ico − Ien‖²₂ / (CHW)`
  - message : `LM = ‖Min − Mout‖²₂ / L` (L2, pas BCE)
  - adverse : `LG = log(1 − A(Ien))` ; discriminateur : `LA = log(1 − A(Ico)) + log(A(Ien))`
  - objectif encodeur/décodeur : `E[LM + λI·LI + λG·LG]` ; discriminateur : `E[LA]` (entraînement alterné type GAN).
- **Couches de bruit** (une par mini-batch pour le modèle « Combined ») :
  - Identity ;
  - Dropout p (garde une fraction p des pixels de `Ien`, le reste vient de `Ico`) ;
  - Cropout p (pareil, en carré) ;
  - Crop p (fraction de surface conservée, jusqu'à 3,5 %) ;
  - Gaussian σ ;
  - JPEG Q.
- **JPEG différentiable** : DCT 8×8 réalisée par une convolution stride 8 à poids fixes.
  - **JPEG-Mask** : garde 25 coefficients basse fréquence en Y et 9 en U/V.
  - **JPEG-Drop** : dropout progressif, plus fort sur les hautes fréquences.
  - Entraîné ainsi, le modèle résiste au vrai JPEG Q=50 : 85 % de bit accuracy, contre 50 % sans bruit.
  - (Pixel Seal remplace cette astuce par un *straight-through estimator*.)

## Architecture (annexe A)
- Encodeur : 4 blocs Conv-BN-ReLU (64 filtres, 3×3). Le message est **répliqué spatialement** (volume L×H×W) puis concaténé avec les features et l'image : (64+L+C)×H×W. Suivent 1 bloc Conv-BN-ReLU de 64 filtres et une conv 1×1 vers C canaux, sans activation.
- Décodeur : 7 blocs Conv-BN-ReLU de 64 filtres, puis 1 bloc à L filtres, **average pooling global** (indépendant de la taille de l'image → robuste au crop) et une couche linéaire L×L. Les bits sont arrondis à 0/1.
- Adversaire : 3 blocs Conv-BN-ReLU, pooling global, linéaire vers 2 classes.
- Entraînement :
  - données : 10k images COCO, test sur 1000 images ;
  - optimisation : Adam lr 1e-3, batch 12 ;
  - durée : 200 époques, 400 si plusieurs bruits.

## Réglages
- **Watermarking** : images YUV 3×128×128, **L = 30 bits**, λI = 0,7, λG = 0,001.
- **Stéganographie** : 52 bits sur des patches gris 16×16, soit 0,203 bpp. Bruit = identité.

## Résultats clés
- **Stéganographie (BOSS 512×512)** :
  - erreur de bits < 1e-5 ;
  - détection par le stéganalyseur ATS : **98 % si les poids sont connus**, **50 % (hasard) si on ne connaît que l'architecture** (poids inconnus). Classiques à titre de comparaison : HUGO 70 %, WOW et S-UNIWARD 68 %.
  - → La **sécurité repose sur le secret des poids** : un modèle par seed donne un motif différent.
- **PSNR(Y) selon le bruit d'entraînement** :

  | Identity | Dropout | Cropout | Crop | Gaussian | JPEG-mask | JPEG-drop | Combined | Digimarc |
  |---|---|---|---|---|---|---|---|---|
  | 44,63 | 42,52 | 47,24 | 35,20 | 40,55 | 30,09 | 28,79 | 33,55 | 62,12 |

  Le PSNR est bas en robuste (~30-35 dB) : **watermark visible selon les standards actuels** (Pixel Seal ~48-52 dB).
- **Robustesse** :
  - le modèle sans bruit tombe à 50 % (hasard) sur Crop et JPEG ;
  - les modèles spécialisés tiennent ;
  - le modèle Combined est proche des spécialisés (94 % contre 97 % sur Cropout).
- **Adversaire** : sans lui (L2 seul), artefacts visibles ; avec lui, aucun artefact visible.
- **Comparaison avec Digimarc** (≥ 95 % de bit accuracy compté comme décodage réussi, ≤ 90 % comme échec) :
  - HiDDeN gagne sur les bruits spatiaux : Dropout p = 0,1 et Crop p = 0,1, là où Digimarc échoue complètement ;
  - **HiDDeN perd sur les bruits fréquentiels** (JPEG, flou) : rien dans l'architecture n'est pensé pour le fréquentiel.

## Limites / angles hackathon
- Faible capacité (30-48 bits), basse résolution (128-256 px), PSNR bas, et pas de robustesse géométrique évaluée (rotation, perspective). Les auteurs le listent eux-mêmes en travaux futurs.
- Watermark global, sans localisation : il s'effondre sous le splicing.
  - D'après la slide organisateur (WAM, COCO) : **HiDDeN détecte < 1 % des images splicées**.
- C'est la **baseline la plus faible** du tableau Pixel Seal. Elle est utile comme point bas d'une courbe d'attaque, et comme preuve que « la recette » marche avec un modèle minuscule (encodeur 0,75 MB en TorchScript).
- Idée pour la démo (inférence) : montrer l'évolution 2018 → 2025 sur la même attaque, HiDDeN → TrustMark → Video Seal → Pixel Seal → WAM, avec le bit accuracy et le PSNR de chacun.
