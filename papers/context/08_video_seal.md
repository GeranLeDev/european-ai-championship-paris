# CONTEXTE AGENT — Video Seal (Meta, arXiv 2412.09492)
Source vérifiée : texte complet lu. Code : https://github.com/facebookresearch/videoseal
Auteurs : Pierre Fernandez*, Hady Elsahar*, I. Zeki Yalniz, Alexandre Mourachko (Meta FAIR, déc. 2024). Démo : https://aidemos.meta.com/videoseal. Prédécesseur de Pixel Seal (`01_pixel_seal.md`).

## En une phrase
Watermarking invisible **post-hoc** multi-bit (96 bits dans le papier) pour vidéos ET images : un embedder U-Net ajoute un signal, un extractor ViT relit le message après édition/codecs ; astuce clé = **temporal watermark propagation** (on n'embarque qu'une frame sur k et on copie le signal) pour être rapide.

## Pipeline & formules
- Embedding haute résolution (Eq. 1) : `xw = αw · resize(w) + x`, `w = Embedder(resize(x), m)`, frame réduite en 256×256 (bilinéaire), `w` ré-agrandi à la résolution d'origine. `αw` réglable à l'inférence (qualité vs robustesse) ; "nominal" = valeur d'entraînement. Extraction : frames redimensionnées en 256×256.
- **Propagation temporelle (Eq. 2)** : `wi = Embedder(xi, m)` si `i mod k = 0`, sinon `wi = w_{i−1}`. k=4 dans tout le papier (k=1 = frame par frame). Entièrement différentiable (utilisée aussi à l'entraînement). Coût : bruit à l'extraction + artefacts "shadow"/clignotement si beaucoup de mouvement ; la robustesse varie peu avec k (ablation Fig. 9).
- **Extraction vidéo (Eq. 3)** : extractor frame par frame, moyenne des soft messages sur T frames, seuil à 0 sur les logits : `m̂k = 1 si (1/T)Σ m̃i,k > 0`. Eval : extraction sur les 3 premières secondes. Plus de frames = meilleure bit acc. (Fig. 9). Variantes d'agrégation (L1/L2/carré) : gain non significatif (ex. combiné 0.730 → 0.742).
- **Pas de JND dans le papier** (αw constant, pas d'atténuation perceptuelle ; seul WAM utilise un heatmap, cité comme lent). Le JND (`modules/jnd.py`) n'apparaît que dans la carte du repo v1.0 (voir plus bas).
- Pertes (Eq. 4–7) : `L = λdisc·Ldisc + λi·Li + λw·Lw`. `Lw` = BCE moyenne sur les bits ; `Li` = **MSE(x, xw)** (LPIPS/Watson testés, gain jugé insuffisant) ; `Ldisc = −Dq(xw)`, discriminateur patch (Isola/Rombach), dual-hinge `½(max(0,1−Dq(x)) + max(0,1+Dq(xw)))`. Poids : λw=1.0, λi=0.5, λdisc=0.1. **Balancer** adaptatif (type EnCodec) : chaque perte est renormalisée par la norme de son gradient sur la dernière couche de l'embedder, R=1.
- Multistage : (1) pré-entraînement **images seules** ; (2) **hybride** images/vidéos en alternance (ablation 1:1) ; (3) **embedder gelé + fine-tuning de l'extractor seul** (gain de bit acc. sans changer le PSNR car le watermark ne change pas).
- Codecs non différentiables (H.264/H.265 via PyAV/FFmpeg) : **straight-through estimator** `xaug = xw + nograd(T(xw) − xw)`.
- Augmentations d'entraînement (Tab. 3), même tirage appliqué à tout le mini-batch : brightness 0.5–2.0, contrast 0.5–2.0, hue −0.5…0.5, saturation 0.5–2.0, flou gaussien k impair 3–17, médian k 3–7, JPEG Q 40–80, **H.264 CRF 9–27**, flip horizontal, crop ratio d'arête 0.7–1.0, resize 0.7–1.5, rotation ±10°, perspective 0.1–0.5. Crop/resize : arêtes indépendantes (le ratio d'aspect peut changer).
- Détection : p-value (Eq. 10) = P(bit acc. ≥ observée | bits Bernoulli(0.5)) = somme binomiale / 2^nbits ; on rapporte log10(p) (FPR < 1e-6 ⇔ log10 p < −6). Capacité de Shannon (annexe A.1) : permet de comparer 32/48/96/256 bits. Le papier précise que c'est un outil conceptuel de comparaison, pas un vrai test d'hypothèse.

## Implémentation
- Embedder : U-Net réduit "Efficient U-Net" (Imagen), **16M params**, entrée 3×256×256, bottleneck 32×32 (dz=128, dmsg=192), DBlock = downsample bilinéaire + ResNetBlock, UBlock = upsample bilinéaire (pas de ConvTranspose, évite le damier), RMSNorm + SiLU. Message : table de lookup `T ∈ (nbits, 2, dmsg)`, embeddings moyennés puis répétés en 32×32 et concaténés au latent.
- Extractor : **ViT-Small** (patch 16, d=384, 22M ; **24M total** avec patch decoder CNN + average pooling + linéaire → nbits, sigmoïde seulement à l'inférence).
- **nbits = 96**, αw = 2.0 en évaluation (sauf mention contraire).
- Données : SA-1B 500k images en 256×256 ; SA-V 51k vidéos (240p→4K, ~14 s, 24 fps), clips de 32 frames (1.3 s) en 256×256. Éval : 1k images SA-1B (~1500×2250) + 1k COCO ; vidéos SA-V = 5 premières secondes en résolution native.
- Entraînement : 16 GPUs, AdamW. 800 epochs images (batch 16/GPU, 1500 steps/epoch, lr 1e-6→1e-5 sur 50 epochs puis cosinus → 1e-7) ; 300 derniers epochs avec SA-V (200 steps/epoch, 1 clip de 32 frames/GPU, lr 1e-7→1e-6 puis cosinus → 1e-8) ; embedder gelé à l'epoch 250.
- **Repo local** (`repos/videoseal/videoseal/cards/`, README : `videoseal.load("videoseal")` = v1.0, 256 bits, stable ; défaut `DEFAULT_CARD='videoseal_1.0'`) :
  - `videoseal_0.0.yaml` = **le modèle du papier** (96 bits, `unet_small2` + extractor `sam_small`, `attenuation: None`, additive, MSE, `lambda_i 0.5 / lambda_d 0.1`, chunk 32 / step 4, finetune extractor dès l'epoch 251). Charger via le nom de carte `videoseal_0.0` (nom exact non confirmé dans le README, qui ne liste que `videoseal`, `pixelseal`, `chunkyseal` ; vérifier `videoseal.load`).
  - `videoseal_1.0.yaml` = version **non décrite dans le papier** : 256 bits, atténuation `jnd_1_1`, `scaling_w: 0.2`, embedder `unet_small2_yuv_quant` (entrée/sortie 1 canal, donc YUV, avec quantification), extractor `convnext_tiny` ; le README la dit à JND heatmap par défaut. Le papier cite YUV/YCbCr uniquement comme piste future (inférence : v1.0 l'implémente).
- Poids v0.0 : `huggingface.co/facebook/video_seal/checkpoint.pth` ; v1.0 : `dl.fbaipublicfiles.com/videoseal/y_256b_img.pth`.

## Résultats clés (papier, modèle 96 bits)
Robustesse SA-V vidéo, moyenne par catégorie (bit acc. / log10 p), k=4, extraction 3 s :
| | HiDDeN | MBRS | CIN | TrustMark | WAM | Video Seal |
|---|---|---|---|---|---|---|
| bits | 48 | 256 | 30 | 100 | 32 | 96 |
| Identity | 0.99/-14.0 | 1.00/-77.1 | 1.00/-9.0 | 1.00/-30.1 | 1.00/-9.6 | 0.99/-26.8 |
| Valuemetric | 0.88/-9.1 | 0.89/-54.3 | 0.93/-7.5 | 0.93/-24.7 | 0.92/-7.7 | 0.90/-19.9 |
| Géométrique | 0.68/-2.9 | 0.50/-0.4 | 0.50/-0.4 | 0.60/-5.5 | 0.81/-5.5 | **0.85/-17.0** |
| Compression | 0.83/-7.2 | 0.79/-34.6 | 0.90/-6.7 | 0.87/-20.0 | 0.86/-6.1 | 0.85/-15.7 |
| Combiné | 0.61/-1.3 | 0.50/-0.4 | 0.49/-0.4 | 0.51/-0.5 | 0.55/-0.8 | **0.73/-8.1** |
(Images SA-1B : Video Seal géométrique 0.83/-16.4, combiné 0.91/-18.4 ; WAM 0.81 et 0.86.)
- Combiné H.264 CRF30 + crop 50 % surface + brightness 0.5 : Video Seal log10 p = −6.1 en moyenne (détectable à FPR < 1e-6).
- Détail SA-V : H264 CRF30 0.97, CRF40 0.89, **CRF50 0.72, CRF60 0.70** ; H265 30 : 0.96, 40 : 0.76, **50 : 0.54 (≈ hasard)** ; (H264 + crop 0.71 + bright 0.5) : CRF30 0.89, CRF40 0.72, CRF50 0.58. Crop 0.55 : 0.80 ; crop 0.71 : 0.97 ; rotation 10° : 0.94 ; perspective 0.5 : 0.94 ; brightness 0.5 : 0.99 ; hue 0.25 : **0.61** (point faible).
- Images SA-1B (détails) : crop 0.32 → 0.50, crop 0.45 → 0.52, crop 0.55 → 0.70 ; rotation 30° → 0.58, 45°/90° → ≈0.5 ; hue ±0.3 à ±0.5 → 0.50 ; JPEG 40 : 0.99 ; (JPEG40 + crop 0.71 + bright 0.5) : 0.89.
- Les baselines MBRS et CIN (crop = masque noir, aucune augmentation géométrique) s'effondrent sur le géométrique. MBRS/TrustMark ont un log10 p plus fort en non-combiné car plus de bits.
- Imperceptibilité (nominal), vidéo SA-V : Video Seal **PSNR 48.02, SSIM 0.998, LPIPS 0.013, VMAF 93.77** ; MBRS VMAF 94.10 (meilleur), TrustMark LPIPS 0.003 (meilleur), HiDDeN PSNR 30.19 / VMAF 74.61. Image SA-1B : Video Seal PSNR 47.39, SSIM 0.999, LPIPS 0.009 ; TrustMark 42.09 / 0.002 ; WAM 39.86 / 0.031. Les auteurs insistent : PSNR/SSIM/VMAF élevés n'empêchent pas des artefacts visibles (zones plates comme le ciel ; "blobs" qui ne suivent pas les objets, contrairement à TrustMark qui produit des ombres suivant le mouvement).
- Vitesse (1 s de vidéo, V100 / 2 Xeon 6230, TorchScript) : Video Seal embed 42.0 GFlops, 1.14 s CPU, 0.42 s GPU ; extract **3.1 GFlops, 0.69 s CPU, 0.11 s GPU** (égal à TrustMark en GPU, 0.11 s). WAM : embed 3.19 s GPU (heatmap calculé sur CPU), extract 68.7 GFlops. Le goulot est le transfert CPU↔GPU et les opérations haute résolution.
- Ablation (nbits=16) : pré-entraînement image + vidéo avec compression = meilleur ; vidéo seule sans pré-entraînement = bit acc. très basse ; fine-tuning de l'extractor = gains sans perte de PSNR ; il vaut mieux garder les compressions dès le début.
- Compromis qualité/robustesse (variation de αw, Fig. 6) : Video Seal domine surtout à PSNR > 50 dB / VMAF > 94 sous combinaison géométrie + codec.

## Différences vs Pixel Seal
- Video Seal : MSE + adversarial (`Li` MSE, λi=0.5), pas de JND dans le papier, αw constant (2.0), 96 bits, embedder 16M + ViT-S 24M, 256×256 pour tout (haute résolution = resize), ~1100 epochs en 3 étapes (images → hybride → finetune extractor). Pixel Seal : 256 bits, perte adversariale seule + JND + boosting, 600k steps, U-Net 43.8M + ConvNeXt-v2 33.4M, entraînement haute résolution.
- Vidéo : Video Seal **entraîne** avec propagation (k=4) et codecs H.264 (STE) ; Pixel Seal n'applique son pooling temporel qu'à l'inférence.
- Pixel Seal rapporte aussi Video Seal 1.0 (256 b) comme baseline : géométrique 0.83, combiné 0.69 sur images Meta AI.

## Limites / angles hackathon
- Limites reconnues : métriques d'imperceptibilité vidéo insuffisantes ; artefacts temporels (shadow/glitter) ; propagation simple ; embedding en RGB (YUV/YCbCr proposé en travaux futurs) ; payload et robustesse à augmenter ; **sécurité du framework non explorée** (cité en travaux futurs).
- Aucune évaluation contre purification/régénération par diffusion ni attaques adversariales sur l'extractor (inférence : angles d'attaque ouverts). Citation : JigMark (diffusion purification) dans l'état de l'art.
- Points faibles mesurés à exploiter : H.264 CRF ≥ 50 (0.72), H.265 CRF 50 (0.54), rotation ≥ 30°, crop < 0.55 d'arête, décalage de teinte ±0.3 (≈0.5), combinaison géométrie + codec fort.
- Piste simple (inférence) : le message est moyenné sur les frames ; attaquer la cohérence temporelle (décaler/rééchantillonner les frames, changer le framerate) et exploiter le fait que le watermark est un motif copié sur k=4 frames (inférence : estimation/retrait par soustraction de la composante statique).
- Défense : agréger plus de frames T à l'extraction, αw plus élevé (au prix du PSNR), détection par `log10 p` plutôt que bit acc.
