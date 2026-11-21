# CONTEXTE AGENT — Watermark Anything (WAM, Meta, ICLR 2025, arXiv 2411.07231)
Source vérifiée : texte complet lu. Code : https://github.com/facebookresearch/watermark-anything · démo HF : https://huggingface.co/spaces/xiaoyao9184/watermark-anything

## En une phrase
Watermarking invisible **post-hoc** vu comme une **segmentation** : l'embedder cache un message de 32 bits, l'extractor sort pour CHAQUE pixel un score « marqué ? » + un message de 32 bits ; on en déduit détection globale, **localisation des zones marquées** (splicing/inpainting) et **plusieurs messages** dans la même image. Co-auteurs : Tom Sander, Pierre Fernandez (Meta), Durmus, Furon, Douze.

## Pipeline & formules
- **Embedder** `xm = x + α·δθ(x,m)`, δ ∈ [-1,1] (tanh en sortie). Autoencoder type LDM/VQGAN : encoder → latent z (dz=4, 32×32) ; chaque bit k de valeur m_k indexe une table `Tθ ∈ R^{nbits×2×dmsg}` ; les nbits embeddings sont **moyennés**, répétés en (dmsg,32,32) et concaténés à z ; le decoder sort δ.
- **Extractor** `y = ext(x) ∈ R^{(1+nbits)×h×w}` = masque de détection `ydet` + message `ydec` par pixel (ViT + décodeur pixel, façon SETR/SAM).
- **Haute résolution** : image redimensionnée en 256×256 (anisotrope), δ calculé en basse rés puis interpolé (bilinéaire) à la taille d'origine ; l'extraction redimensionne aussi en 256×256. Entraîné seulement en basse rés.
- **Augmenter (2 étapes)** : (1) splicing par masque aléatoire `xmasked = rmask⊙xm + (1−rmask)⊙x` (masques : pleins, rectangles, irréguliers, segmentation COCO ; inversés avec proba 0.5) ; (2) transformations : géométriques (identity, resize, crop, rotate, hflip, perspective) + valuemetric (JPEG, blur gaussien, médian, brightness, contrast, saturation, hue). Le masque subit la même géométrie → vérité terrain `ydet,⋆`.
- **Pertes phase 1** : `ℓ = λdet·ℓdet + λdec·ℓdec`. ℓdet = BCE par pixel (marqué ou non). ℓdec = BCE par pixel et par bit, **uniquement sur les pixels marqués**. Pas de perte perceptuelle.
- **Phase 2 (post-training)** : (a) carte **JND** (Chou & Li 1995 : luminance masking + contrast masking, canaux (R,G,B)=(1,1,2)) : `xm = x + αJND·JND(x)⊙δ` ; (b) **multi-messages** : 1 à 3 masques disjoints (proba 0.6/0.2/0.2) avec messages aléatoires distincts ; ℓdet prend l'union des masques, ℓdec calculée par message puis sommée. Sans phase 2, le modèle prédit le même message partout.
- **Inférence** : seuil pixel `τ` (calibré sur le FPR pixel d'un set d'entraînement tenu à part ; τ=0.5 pour le multi-message).
  - Détection image : `sdet = (1/hw)·Σ 1{ydet_i > τ}` ; image marquée si sdet > seuil (0.07 dans leur éval, ≈ même FPR que les baselines).
  - Décodage 1 message : moyenne pondérée sur les pixels détectés, `m̂k = 1[ (1/Σ1{ydet>τ}) · Σ 1{ydet_i>τ}·ydec_i,k > 0.5 ]` (vote majoritaire soft).
  - **Multi-messages** : on garde les pixels `ydet>τ`, on binarise `m̃_i,k = 1{ydec_i,k > 0.5}`, puis **DBSCAN** (scikit-learn, distance = écart de bits) sur ces messages locaux. Pas besoin de connaître le nombre de messages ; les centroïdes sont des mots binaires réels. Paramètres éval : **ε=1, minsamples=1000 px (~2 % de l'image)**.

## Implémentation
- Embedder **1.1M** params (encoder 4 blocs résiduels, canaux 32,32,32,64, downsampling ×2 sur 3 blocs → f=8, Up par interpolation + conv 3×3, pas de ConvTranspose à cause des artefacts checkerboard). Extractor **96M** (ViT-Base 86M, patch 16, d=768 ; décodeur pixel 3 blocs Up ×4,×2,×2 + Linear → 1+nbits). nbits = dmsg = 32. L'embedder est volontairement petit (rapide côté utilisateur).
- Données : MS-COCO train (118k images, visages floutés), résolution 256×256.
- Phase 1 : AdamW, warmup linéaire 5 epochs 1e-6 → 1e-4 puis cosine → 1e-6, **λdec=10, λdet=1, α=0.3**, batch 16/GPU, **300 epochs**, 8 V100, ~2 jours. Masques phase 1 : irrégulier/boîte/plein/segmentation à 0.25 chacun.
- Phase 2 : **200 epochs**, JND avec **αJND=2**, jusqu'à 3 messages par image (rectangles ou masques de segmentation). Coût total ≈ 30 GPU-jours.
- Paramètres d'entraînement des augmentations : JPEG Q 40–80, crop ratio de côté 0.33–1.0, rotation ±10°, perspective 0.1–0.5, brightness/contrast/saturation 0.5–2.0, blur 3–17.

## Résultats clés
**Imperceptibilité (Tab. 1)** : COCO PSNR 38.3 / SSIM 0.99 / LPIPS 0.04 ; DIV2K PSNR 38.8 / SSIM 0.99 / LPIPS 0.03 (TrustMark : 40.3 / 0.01 sur COCO). Filigrane parfois visible (fourrures, zones très blanches/sombres, motifs répétitifs).

**Tab. 2a — COCO (10k imgs), TPR % / Bit acc % (FPR WAM 0.04 %)**
| Méthode | None | Géométrique | Valuemetric | Inpainting (LaMa) | Splicing |
|---|---|---|---|---|---|
| HiDDeN | 76.9/95.5 | 31.2/80.1 | 48.4/87.2 | 44.7/88.7 | 0.9/68.0 |
| SSL | 99.9/100 | 14.3/76.5 | 70.5/92.1 | 67.7/91.1 | 0.4/58.9 |
| FNNS | 99.6/99.9 | 62.4/86.6 | 82.1/93.9 | 89.4/97.3 | 38.7/88.5 |
| TrustMark (FPR 2.88) | 99.8/99.9 | 36.3/71.4 | 90.1/98.2 | 39.4/83.2 | 83.2/57.1 |
| **WAM** | 100/100 | 99.3/91.8 | 100/99.9 | 97.9/99.2 | 100/95.3 |

**Tab. 2b — DIV2K haute rés (100 imgs), WAM** : None 100/99.9 ; Géom 96.1/89.0 ; Value 100/99.9 ; Inpainting 100/99.8 ; Splicing 99.5/94.2 (TrustMark inpainting 0.0/72.1, splicing 1.5/57.3).

**Tab. 7 (WAM, TPR / Bit acc, COCO)** : crop 20 % 99.7/88.8 ; crop 50 % 99.6/95.9 ; rotation 10 97.7/77.0 ; perspective 0.5 100/96.0 ; JPEG 50 99.9/99.0 ; blur 17 100/99.8 ; proportion 10 % 99.9/94.2 ; collage 10 % 100/96.5 ; **combinaison JPEG80+brightness 1.5+crop 50 % : 99.2/87.2**. Points bas notables : rotation 10 (bit acc 77.0), crop 20 % (88.8), combinaison (87.2).

**Localisation (Fig. 4, COCO 512×512, vs EditGuard)** : bon mIoU même après crop 25 % + resize, sauf quand la zone marquée couvre 95 % (tout est classé marqué). Bit acc : ~31/32 bits avec 10 % d'une image 256×256 marquée ; ~25 bits avec 10 % d'un crop 25 % (2.5 % des pixels). EditGuard non robuste au crop.

**Multi-messages (Fig. 5, 1 à 5 messages de 32 bits, zones 10 % non chevauchantes)**
| | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Après phase 1 : bit acc | 0.95 | 0.78 | 0.67 | 0.62 | 0.62 |
| Après phase 1 : clusters DBSCAN | 1.00 | 1.03 | 1.01 | 1.09 | 1.05 |
| Après phase 2 : bit acc | 0.94 | 0.97 | 0.98 | 0.98 | 0.98 |
| Après phase 2 : clusters DBSCAN | 1.00 | 2.00 | 2.99 | 3.99 | 4.95 |
mIoU phase 2 = 0.98 partout. Après flip + contraste 1.5 : mIoU 85 %, bit acc > 95 % (160 bits). **Avec JPEG-80 en plus : mIoU 59 %, bit acc ≈ 0.60, nombre de clusters chute (ex. 4.64 → 1.31 selon ε/minsamples)** = échec. Overlap total de 2 watermarks : détection 100/100 (Tab. 11).

**Générateurs (Tab. 6, AUC / TPR@1e-4)** : sous splicing, WAM TPR@1e-2 = 1.00 mais TPR@1e-4 = **0.00** ; Tree-Ring 0.00/0.00 ; Stable Signature 0.90/0.81. Combiné : WAM AUC 1.00.

**Tab. 8 (bit acc, MBRS/CIN/WAM)** : géométrique 50.4 / 50.3 / 91.8 ; splicing 65.31 / 96.47 / 95.30.

**Purification (Tab. 9, bit acc WAM)** : DiffPure PSNR 30.1 → 100.0 ; 28.4 → 99.4 ; 27.0 → 71.3 ; 24.9 → 49.1. Attaque VAE PSNR 32.9 → 100.0 ; 32.4 → 99.8 ; 28.6 → 98.6 ; 25.5 → 53.7. (TrustMark/MBRS/CIN tiennent mieux à 27.0 / 25.5 mais sont fragiles en géométrique.)

**Compromis robustesse/visibilité (Tab. 10, αJND à l'inférence ; JPEG80 + crop 50 % + brightness 1.5)** : 1.0 → PSNR 44.1 / bit acc 71.0 ; 1.5 → 40.7 / 84.3 ; 2.0 → 38.3 / 88.8 ; 2.5 → 36.4 / 90.8 ; 3.0 → 34.8 / 91.9.

## Baseline dans le repo `videoseal`
`repos/videoseal` peut charger WAM comme baseline via `build_baseline('wam')` avec `ckpts/wam_encoder.pt` et `ckpts/wam_decoder.pt` (déjà téléchargés, TorchScript, 32 bits). (Inférence : cette interface baseline donne probablement un message global, pas la carte de localisation par pixel ; pour DBSCAN/masque, utiliser le repo officiel `watermark-anything`.)

## Limites / angles hackathon
- **Payload faible : 32 bits** (entraîner plus gros est dit « challenging »). Détection séparée du décodage, donc pas de garantie p-value comme Pixel Seal.
- **Fragilité mesurée (angles d'attaque)** : multi-messages + JPEG-80 (échec), purification diffusion (bit acc 49.1 à PSNR 24.9), VAE à PSNR ≈ 25 dB (53.7), rotation 10° (77.0), crop 20 % (88.8). Un grand watermark marquant 95 % de l'image fait tout classer « marqué ».
- **Forge / faux positif** : le filigrane peut être visible (motifs répétitifs, fourrure, zones claires/sombres) ; JND ne tient compte que de l'image, pas du signal δ. (Inférence : cette régularité pourrait faciliter l'estimation du motif et la suppression ou la forge ; non testé dans le papier.)
- **Seuil de détection** : sdet > 0.07 est empirique ; le papier ne fournit pas de p-value, et TPR@1e-4 s'effondre sur splicing (Tab. 6). Aucune attaque adversariale ciblée sur l'extractor n'est évaluée (inférence : probablement vulnérable, à tester).
- **Éval du papier** : DiffPure et VAE seulement dans l'appendice G ; pas d'attaque white-box.
- **Idées** : (attaque) régénération diffusion ou VAE à ~25 dB, JPEG-80 sur image multi-marquée pour casser le comptage de messages, collage d'un grand watermark pour forger ; (défense/produit) carte de localisation pour détecter le splicing/inpainting, `αJND` à 2.5–3.0 pour renforcer la robustesse ; comparer WAM et Pixel Seal (`01_pixel_seal.md`) sur les mêmes attaques via le repo `videoseal`.
