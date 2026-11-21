# CONTEXTE AGENT — The Stable Signature (Meta AI / Inria, arXiv 2303.15435, ICCV 2023)
Source vérifiée : texte complet lu. Code : github.com/facebookresearch/stable_signature · Auteurs : Fernandez, Couairon, Jégou, Douze, Furon (Pierre Fernandez = organisateur).

## En une phrase
Watermark **in-model** (généré à la source) : on fine-tune le **décodeur latent** d'un LDM (Stable Diffusion) pour que TOUTES ses images contiennent une signature binaire k bits, relue par un extracteur fixe pré-entraîné + test statistique. Pas de changement d'architecture ni du processus de diffusion → compatible text-to-image, inpainting, édition, super-résolution.

## Méthode & formules
**Phase 1 — extracteur W (HiDDeN simplifié).** Encodeur WE + extracteur W entraînés ensemble ; WE jeté après. `xw = xo + α·δ` (δ = résidu, tanh sur la sortie de WE + facteur α, pas de loss perceptuelle ni adversaire car WE est jeté). Transformation T aléatoire (crop, resize, JPEG…) entre les deux. `Lm = BCE(m, σ(W(T(xw))))`. Décodage = signe des composantes.
- **Whitening PCA** en fin de pré-entraînement : sans lui les bits de W sur images vanilla sont corrélés et biaisés (un bit à 0.95 de moyenne), ce qui casse le test statistique. PCA sur 10k images vanilla : moyenne µ, `Σ = UΛUᵀ`, couche linéaire de poids `Λ^{-1/2}Uᵀ` et biais `−Λ^{-1/2}Uᵀµ`. Après whitening le max de moyenne d'un bit = 0.68 (simulation Bernoulli : 0.52) : pas parfaitement i.i.d. mais assez proche.
- JPEG non différentiable : `x' = xaug + nograd(xaug,JPEG − xaug)`.

**Phase 2 — fine-tuning du décodeur D → D_m.** Signature m fixe. `z = E(x)`, `x' = D_m(z)`, `m' = W(x')`. `Lm = BCE(σ(m'), m)`. `Li` = perte perceptuelle **Watson-VGG** entre x' et `x'_o = D(z)` (décodeur original) ; c'est elle qui apprend le masquage luminance/contraste (watermark dans les zones texturées, fonds uniformes intacts). **`L = Lm + λi·Li`, λi = 2.0** par défaut.

**Test statistique (détection).** H0 : bits de m' i.i.d. Bernoulli(0.5) → M(m,m') (bits qui matchent) ~ Binomiale(k, 0.5). Flag si `M ≥ τ`. `FPR(τ) = I_{1/2}(τ+1, k−τ)` (bêta incomplète régularisée).
**Identification** : N utilisateurs, chacun son modèle/signature ; on prend `argmax_i M(m', m(i))`. `FPR(τ,N) = 1 − (1−FPR(τ))^N ≈ N·FPR(τ)`. Plus N est grand, plus τ doit monter → baisse de la détection (compromis détection vs traçabilité).

## Implémentation (chiffres du texte)
- Extracteur : architecture HiDDeN (encodeur 4 blocs Conv-BN-ReLU 64 filtres ; extracteur 7 blocs + bloc k filtres + avg pool + linéaire k×k). k = 48 bits, α = 0.3, MS-COCO 256×256, 300 epochs, 8 GPU, Lamb, ~1 jour. Attack layer : crop ou resize (0.3 ou 0.7) / identité, puis JPEG (50 ou 80) avec proba 0.5.
- Fine-tuning décodeur : **100 itérations**, AdamW, batch 4 (<500 images), **~1 minute sur 1 GPU**, lr warmup 20 it. à 1e-4 puis cosine → 1e-6.
- LDM de l'étude : diffusion 2.2B params, 330M paires image-texte internes ; auto-encodeur KL f=8 (celui de SD).

## Résultats clés
- **Détection (text-to-image, 48 bits, FPR inféré par la formule, validé empiriquement jusqu'à ~1e-7 sur 1.4M images ImageNet ; FPR théorique légèrement supérieur à l'empirique sous 1e-6)** : image non modifiée → 99 % détectées à FPR 1e-9 ; à FPR 1e-9 : crop 10 % → 84 %, combiné (crop 50 % + brightness 1.5 + JPEG 80) → 65 %. Abstract : crop 10 %, >90 % d'accuracy à FPR < 1e-6. (Le texte ne cite pas 1e-10 ; seuls 1e-6 et 1e-9.) Forensique passive [Corvi] : ~50 % TPR pour FPR ~1/100.
- **Identification (FPR 1e-6, τ=41 pour N=1, τ=44 pour N=1000)** : 1000 modèles × 100 images ; N=10^5 utilisateurs : 98 % sans modif, 40 % en combiné (« 3 images → 80 % d'identification »). Fausse accusation = 0 à cette échelle.
- **Bit accuracy (Table 1, 512×512)** — None / Crop / Bright / Combiné : text-to-image 0.99/0.95/0.97/0.92 ; DiffEdit 0.99/0.95/0.98/0.94 ; inpainting full 0.99/0.97/0.98/0.93 ; **inpainting masque seul 0.89/0.76/0.84/0.78** (le point faible : signal présent seulement dans la zone remplie) ; super-res 0.98/0.93/0.96/0.92. Table 2 : JPEG 50 = 0.88, resize 0.7 = 0.91, med. filter k=7 = 0.94, rotation 90 non entraînée = 0.56 sans attack layer (0.99 avec).
- **Qualité** : PSNR ~30 dB / SSIM ~0.89 entre générations original vs watermarké (non optimisé), FID inchangé (−0.3 en text-to-image). Coût d'embedding **0.00 s/img** vs 0.11 (HiDDeN) à 0.45 s/img (SSL) en post-hoc. DCT-DWT (celui de SD) : 0.86 / 0.52 / 0.51 / 0.51.
- **Compromis qualité/robustesse (λi)** : 0.8→0.025 : PSNR 31.4→24.6, bit acc combiné 0.85→0.95.
- La robustesse vient de l'**extracteur** (attack layer du pré-entraînement) : crop/brightness gratuits, rotation et JPEG exigent d'être simulés pendant l'entraînement.

## Attaques (section 7)
- **Suppression image-level** : auto-encodeurs neuronaux (Cheng2020, Bmshj2018, VQ-VAE, KL-VAE) : pour ramener la bit acc près de 50 % il faut PSNR < 26 ; **mais si l'attaquant a le même auto-encodeur KL-f8 que le LDM → watermark effacé avec PSNR > 29** (hypothèse forte, Alice n'a pas à distribuer le générateur original).
- **White-box (extracteur fuité)** : attaque adversariale (10 it. Adam lr 0.1, MSE vers un message aléatoire) efface à PSNR plus bas ; permet aussi l'**embedding non autorisé** (usurper l'identité d'un autre utilisateur) → garder l'extracteur privé.
- **Model purification** (fine-tuner le décodeur sans Lm, avec perte image seule) : difficile de baisser la bit acc sans artefacts visibles.
- **Quantization/pruning du décodeur** : 8/4 bits → 0.99 ; pruning L1 30 % → 0.99, 60 % → 0.95 ; la qualité se dégrade avant le watermark.
- **Collusion (moyenne de poids, type model soups)** : bit ℓ = 0/1 si les deux Bob s'accordent, **aléatoire s'ils divergent** (« marking assumption » du traitor tracing vérifiée sans l'avoir visée) ; vaut aussi si la collusion est image-level. Pistes : codes de traçage plus puissants (Tardos).

## Limites & angles hackathon
- Poids du décodeur ouvert = purification/remplacement du décodeur possible ; décodeur de rechange = watermark perdu (inférence, cohérent avec la motivation de DistSeal).
- Extracteur à garder secret (white-box). Bits non strictement i.i.d. (voir whitening).
- Inpainting « masque seul » faible ; résultats d'identification partiellement **extrapolés** (le texte le dit).
- Angles défi : (1) attaque par auto-encodeur du même VAE (KL-f8) = suppression quasi gratuite ; (2) régénération/crop/JPEG fort pour tester le seuil τ ; (3) collusion de deux modèles marqués ; (4) calcul du p-value binomial `scipy.stats.binom.sf` pour un score maison.
