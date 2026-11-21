# CONTEXTE AGENT — EditGuard (Peking Univ., arXiv 2312.08883, CVPR 2024)
Source vérifiée : texte complet lu (papers/txt/editguard.txt, supplément inclus). Papier de COMPLÉMENT : baseline de **localisation de falsification** à laquelle WAM se compare (slide organisateurs : la localisation d'EditGuard tombe à zéro une fois l'image croppée — non présent dans le papier, voir limites).

## En une phrase
Watermarking **proactif** à double charge : un watermark 2D de **localisation** (image cachée, semi-fragile) + un watermark 1D de **copyright** (64 bits, robuste), pour localiser pixel par pixel les zones modifiées (y compris par IA générative) sans jamais entraîner sur des images falsifiées (zero-shot).

## Méthode & formules
- Observation clé : la stéganographie image-dans-image (I2I) est **fragile** (si le container est modifié, l'image secrète révélée est abîmée) et **locale** (les artefacts de l'image révélée correspondent pixel à pixel aux zones modifiées, voisinage immédiat). Donc l'image secrète = watermark de localisation. Le copyright (bits-dans-image) est robuste par nature.
- Cadre « **sequential encoding, parallel decoding** » : encodeur double = IHM (Image Hiding Module) puis BEM (Bit Encryption Module), séquentiels → `Icon`. Décodage en parallèle : tamper locator (IRM + mask extractor) et copyright extractor (BRM). Le codage parallèle (2D + bits reshapés en image 8×8) échoue : bit acc → ~50 % sous JPEG Q=70 (Tab. 11).
- Modèle de transmission : `Irec = D(Icon ⊙ (1−M) + T(Icon) ⊙ M)` (T = falsification, D = dégradation, M = masque).
- IHM/IRM : 6 **blocs inversibles** (flow) sur DWT, couplages affines additifs ; chaque `φ` = dense block à 5 couches + LFIM (module léger d'interaction de features). Images révélée et originale ré-obtenues par le sens inverse.
- **PPEM** (prompt-based posterior estimation) : à la révélation, l'info perdue à la compression est estimée `Ẑ = E[Z|Irec]` (8 blocs résiduels + 8 transformers canal) au lieu d'un bruit gaussien ou zéro ; 3 « degradation prompts » appris (bruit gaussien, Poisson, JPEG) fusionnés via poids softmax. Rend la révélation robuste aux dégradations avec un seul jeu de paramètres.
- BEM/BRM : bits → MLP → cartes L×L, fusionnés dans un U-Net à plusieurs niveaux ; décodage : U-Net → L×L → MLP, seuil 0.
- **Masque** : `M̂[i,j] = 1 si max_c |Ŵloc[i,j,c] − Wloc[i,j,c]| ≥ τ`, **τ = 0.2**. `Wloc` doit être connu aux deux bouts (image solide bleue [0,0,255] à l'inférence).
- **Logique de décision** : (1) `ŵcop ≉ wcop` : image non enregistrée ou fortement altérée globalement, non fiable ; (2) `ŵcop ≈ wcop` et `M̂ ≠ 0` : falsifiée, masque = zones ; (3) `ŵcop ≈ wcop` et `M̂ ≈ 0` : authentique.
- Entraînement **bi-niveau** : (a) BEM+BRM : `ℓcop = ||Icon − Imed||² + λ·||ŵcop − wcop||²`, λ=10 ; (b) on gèle BEM/BRM et on entraîne IHM+IRM : `ℓloc = ||Îori − Iori||₁ + α·||Icon − Iori||² + β·||Ŵloc − Wloc||₁`, α=100, β=1. Seules des dégradations (pas de falsification) sont appliquées à Icon. Sans bi-niveau, bit acc ≈ 49 % (non convergence).

## Implémentation
- COCO train, 250K itérations, Adam (β1=0.9, β2=0.5), lr 1e-4 divisé par 2 tous les 30K, batch 4. Message 64 bits ; une image bleue pure injectée comme Wloc toutes les 10K itérations. Simulateur JPEG différentiable.
- 5.45 M paramètres, 102.7 GFLOPS, 0.069 s en décodage (3×512×512, RTX 3090Ti) : le plus petit modèle comparé.
- Arbitraire résolution (contrairement à MaLP, d'après le papier). Datasets maison **AGE-Set-C** (30K train, 1.2K test : SD Inpaint, ControlNet, SDXL, Lama sur COCO ; RePaint, FaceSwap sur CelebA) et **AGE-Set-F** (100 images retouchées finement via SD-WebUI, Photoshop, Adobe Firefly).

## Résultats clés
- Benchmarks classiques (F1 ; bit acc) : CASIAv1 0.954 / 99.91 ; Coverage 0.955 / 100 ; NIST16 0.911 / 99.88 ; Columbia 0.988 / 99.93. Meilleur passif : TruFor 0.822 / 0.735 / 0.470 / 0.914 (écart de +0.102, +0.116, +0.441, +0.065 en F1).
- AGE-Set-C (F1 / IoU) : SD Inpaint 0.966 / 0.936 ; ControlNet 0.968 / 0.940 ; SDXL 0.965 / 0.936 ; RePaint 0.967 / 0.938 ; Lama 0.965 / 0.934 ; FaceSwap 0.896 / 0.876 ; bit acc ≥ 99.86 %. Passifs : MVSS-Net/OSN/PSCC-Net/IML-ViT F1 ~0.02–0.3 ; HiFi-Net 0.48–0.78 ; MVSS-Net finetuné 0.18–0.69 (oubli catastrophique).
- Qualité (Tab. 2, 128×128, 30 bits) : PSNR 36.93, SSIM 0.944, NIQE 5.567, BA 99.89 (CIN 41.35 PSNR mieux, PIMoG 36.22, SepMark 35.42, MBRS 26.57) ; en 512×512/64 bits : PSNR 37.77, SSIM 0.949, BA 99.95.
- Robustesse (SD Inpaint ; F1 / BA) : propre 0.966 / 99.95 ; gaussien σ=1 0.937 / 99.94, σ=5 0.932 / 99.37 ; JPEG Q=70 0.920 / 97.69, Q=80 0.920 / 98.16, Q=90 0.925 / 98.23 ; Poisson 0.943 / 99.91. Testé aussi sur pipeline « WeChat » (qualitatif).
- Ablation Wloc : bleu pur 0.966 F1 ; rouge 0.946 ; vert 0.885 ; image naturelle 0.851 (seuil τ difficile). Comparaison I2I : HiNet 0.952 propre mais 0.025 sous JPEG Q=80 ; RIIS 0.854 / 0.833.
- Détection stéganalyse (StegExpose) : plus sûr que HiNet/RIIS, SepMark, MBRS ; légèrement moins que CIN.

## Limites & angles hackathon
- **Avoué dans le papier** : choix du Wloc empirique ; robustesse à améliorer ; pas de sémantique ; testé uniquement sur dégradations bruit/JPEG/Poisson (limites : section 10).
- **Non testé dans le papier** : crop, resize, rotation, flou, etc. ; le papier ne traite que Gaussien, JPEG, Poisson. (inférence) la localisation est pixel-alignée avec `Wloc` (même H×W, masque comparé position par position) : un crop/décalage/resize casse l'alignement, d'où l'effondrement à zéro montré par les organisateurs de WAM, qui sait localiser sans alignement.
- Copyright = 64 bits seulement, extracteur entraîné sans augmentations géométriques (inférence : à vérifier par test).
- Angle d'attaque (inférence) : un recadrage ou resize suffit à neutraliser la localisation ; régénération complète (diffusion/VAE) détruit le watermark ; sans le `Wloc`/poids, le masque ne peut pas être calculé (secret partagé nécessaire).
- Intérêt : baseline de localisation à comparer à WAM : même scénario inpainting/AIGC, mesurer F1/IoU avant/après crop.
