# CONTEXTE AGENT — TrustMark (Adobe / Univ. Surrey, arXiv 2311.18297, nov. 2023)
Source vérifiée : texte complet lu (papers/txt/trustmark.txt, supplément inclus). Code : https://github.com/adobe/trustmark (`pip install trustmark`). Papier de COMPLÉMENT (baseline), pas un des 3 papiers cœur. Dans `repos/videoseal`, TrustMark-Q est chargé comme baseline via `build_baseline('trustmark')` (ckpts/trustmark_encoder_q.pt, déjà téléchargé).

## En une phrase
Watermarking invisible **multi-bit** post-hoc, GAN, entraîné à 256×256 puis étendu à **toute résolution** par un « resolution scaling » appliqué à l'inférence ; accompagné d'un réseau de **suppression** de watermark (TrustMark-RM) pour re-watermarker.

## Méthode & formules
- **Resolution scaling (Algo. 1)**, boîte noire, valable pour n'importe quelle méthode : `x ∈ [-1,1]` → `x̄ = resize(x, 256×256)` → `r = E(x̄, w) − x̄` (résidu) → `r ← resize(r, H×W)` → `y = clamp(x + r, −1, 1)`. L'image haute rés. est donc l'original + résidu upsamplé (pas un ré-échantillonnage de l'image). Exige un modèle robuste au resize (couvert par le bruit d'entraînement). Même algo pour RM : `r = R(x̄) − x̄`.
- Contrôle à l'inférence (supp. C) : `y = clamp(x + λ·r, −1, 1)`. λ→0 = plus invisible/moins robuste ; plage raisonnable λ ∈ [0.75, 1.5]. Réglable image par image.
- Embedder E : `Epre(x,w)` (watermark interpolé à 256×256, concaténé, conv 3×3 → d canaux) → réseau basé MUNIT (sans normalisation de canaux, profondeur ×2, dim. interne ÷2) → `Epost` : plusieurs convs **1×1** séparées par SiLU, finissant en tanh. E sort l'image encodée directement (pas un résidu explicite à l'entraînement).
- Extracteur X : **ResNet50**, dernière couche = FC sigmoïde à `l` bits. Seules les familles ResNet convergent (ResNet18, DenseNet121, RegNet, ResNeXt OK ; VGG, GoogleNet, ConvNeXt, EfficientNet échouent) : il faut un chemin résiduel/skip.
- Bruit N (tout différentiable, Kornia ; JPEG différentiable de Shin & Song) : 3 transformations géométriques (flip, crop, resize ; flip p=0.5, resize 0.8–1.0, aspect 3/4–4/3, crop 244/256) + 15 perturbations (JPEG, luminosité, teinte, contraste, netteté, color jitter, RGB shift, saturation, gris, flou gaussien/médian/box/mouvement, bruit gaussien, posterize). Par image : 3 géométriques + 2 perturbations aléatoires.
- Pertes : `Ltotal = α·Lquality + Lrecovery` ; Lrecovery = BCE ; `Lquality = βYUV·MSE_YUV + βLPIPS·LPIPS + βFFL·FFL + βGAN·GAN+GP` (WGAN-GP). **FFL = focal frequency loss** (Fourier 2D, poids dynamiques) utilisée pour préserver les hautes fréquences. Poids : βLPIPS=1, βYUV=1.5, βFFL=1.5, βGAN=1.
- **Curriculum 4 étapes** (déclenchées par seuils de bit accuracy d'entraînement) : étape 0 α=0.05, batch fixe répété (seuls les watermarks varient), sans bruit ni GAN ; 90 % → batchs aléatoires ; 95 % → bruit activé ; 98 % → GAN activé et α monte linéairement vers αmax sur 10 000 itérations. TrustMark-B atteint les étapes 1/2/3 en ~500/600/800 itérations.
- **Variantes** (paramètre αmax, fixé à l'entraînement) : **TrustMark-Q** (qualité, αmax=27.5), **TrustMark-B** (balanced, αmax=20). Plage stable αmax ∈ (0,30). Une variante « P » n'apparaît pas dans le texte du papier (inférence : variante ajoutée dans le repo, à vérifier dans le README du repo).
- **TrustMark-RM** : basé sur KBNet (restauration d'image) ; E et X gelés ; pour chaque image on embarque n=3 watermarks aléatoires ; pertes : MSE (PSNR), GAN+GP, et perte de similarité de réponse de X (sortie de X sur l'image nettoyée ≈ hasard).

## Implémentation
- Données : 101K images MIRFlickR (1K validation) ; test CLIC, DIV2K (900 images 2K), MetFace.
- 150 epochs, AdamW, lr 4e-6 par image, batch 32, cosine ; 48 h sur une RTX 3090. RM : ~2 semaines, 100 epochs, batch 8, A100.
- Inférence (RTX 3090) : encodage 125 ms, décodage 25 ms.
- Taille du message testée : 32 à 200 bits (32→200 : −7.5 dB PSNR, −11 % bit acc). Pas de code correcteur dans les résultats ; BCH suggéré pour corriger les rares erreurs de bits.
- Repo : voir `pip install trustmark` ; le payload par défaut et l'ECC du repo ne sont pas dans le papier (non vérifié).

## Résultats clés
- Tab. 1 (CLIC / DIV2K / MetFace ; PSNR ; bit acc propre → bruitée) :
  - TrustMark-Q : PSNR 43.26 / 42.39 / 45.34 ; acc bruitée 0.95 / 0.95 / 0.96 (propre 1.00).
  - TrustMark-B : PSNR 41.53 / 40.20 / 43.87 ; acc bruitée 0.97 / 0.97 / 0.97.
  - RoSteALS : PSNR 30.03 / 27.95 / 33.77 ; acc bruitée 0.94 / 0.93 / 0.93.
  - RivaGAN : PSNR ~41 ; acc bruitée 0.79 / 0.78 / 0.82. SSL : PSNR ~42.7 ; acc bruitée 0.60 / 0.57 / 0.70. StegaStamp : PSNR 37.48 / 35.87 / 39.35 ; acc bruitée ~0.72. dwtDctSvd : acc bruitée ~0.52 (hasard).
  - DIV2K = benchmark le plus dur. Q ≈ 2 dB de PSNR au-dessus de B ; B gagne 1–2 % d'acc bruitée.
- Tab. 2 (config RoSteALS, bruit ImageNet-C, sans scaling) : TrustMark αmax=15 → PSNR 38.87, acc 0.95 ; RoSteALS 32.68 / 0.94 ; StegaStamp 31.26 / 0.88 ; SSL 41.84 / 0.62 ; RivaGAN 40.32 / 0.77 ; dwtDctSvd 38.96 / 0.61.
- Trade-off αmax : de ~0 à 30, PSNR +8 dB pour <5 % d'acc perdue. Sévérité du bruit d'entraînement : sans bruit PSNR 53.2 dB, bruit fort 40.2 dB, acc −3 % seulement.
- Robustesse par attaque (Fig. 5, valeurs non extraites du texte) : TrustMark-Q > RoSteALS sur tout sauf bruit gaussien et box blur ; dwtDctSvd ≈ hasard sur RGB shift.
- **Attaque adversariale I-FGSM** (ε=8/255, sur l'extracteur) : sans bruit d'entraînement <50 itérations suffisent ; avec bruit fort, 32 % des images demandent >3000 itérations.
- **Suppression (Tab. 3, DIV2K, TrustMark-B)** : TrustMark-RM PSNR 48.48, SSIM 0.997, bit acc tombe à 0.553 ; I-FGSM : PSNR 23.48, acc 0.629. Pas exactement 0.5 car le resolution scaling affaiblit l'effet. Re-watermarking : l'acc n'est pas affectée par RM, la qualité est mieux préservée, mais le bruit de RM s'accumule au fil des passes.
- Ablation (Tab. 4) : GAN+Epost+FFL = +4 dB et +1.4 % d'acc ; sans rien, PSNR ≈ StegaStamp. Resolution scaling : DIV2K de 20 % à 100 % de la résolution, PSNR ±0.02 dB, acc ±1e-4 ; perte <0.1 dB vs chute plus forte en bilinéaire/bicubique.
- Remarque PSNR (supp. B) : le PSNR dépend de la résolution d'évaluation (jusqu'à 4.7 dB d'écart pour une paire cover/watermarked), donc PSNR à résolution native non comparable entre méthodes.

## Limites & angles hackathon
- Limite du papier : images très encombrées (hautes fréquences) → PSNR et acc légèrement plus bas.
- Le résidu est upsamplé : les attaques de resize/recompression font aussi baisser l'acc (le papier montre ~0.95 en moyenne sur le bruit mixte, pas de détail par attaque dans le texte).
- Face aux méthodes Meta (Pixel Seal/WAM) : TrustMark = baseline de bonne qualité visuelle (PSNR >40 dB) mais extracteur ResNet50 à 256×256 → (inférence) moins robuste au crop localisé/inpainting que les extracteurs de Pixel Seal ; test à faire via `build_baseline('trustmark')`.
- Attaque sur mesure possible : TrustMark-RM est un débruiteur dédié au résidu du watermark (angle suppression/white-box), I-FGSM sur l'extracteur (white-box) est efficace si l'entraînement est sans bruit fort.
- Détection : pas de p-value dans le papier (bit acc uniquement) ; on peut appliquer la p-value binomiale de Pixel Seal (inférence).
