# CONTEXTE AGENT — Tree-Rings Watermarks (Wen, Kirchenbauer, Geiping, Goldstein ; U. Maryland, NeurIPS 2023)
Source vérifiée : texte complet lu. Code : github.com/YuxinWenRick/tree-ring-watermark

## En une phrase
Watermark **out-of-model, sans entraînement** : on impose un motif (« clé ») dans l'espace de Fourier du **bruit initial** x_T d'un modèle de diffusion ; la détection **inverse** l'image par DDIM pour retrouver x_T puis teste la présence de la clé. Aucune modification de l'image échantillonnée : « vraiment invisible » (la distribution est légèrement décalée, les mises en page générées changent juste).

## Méthode & formules
- **Embedding** : masque binaire M (disque de rayon r centré sur les basses fréquences, r = 10 par défaut) ; `F(x_T)_i = k*_i` si i ∈ M, sinon `N(0,1)`. Puis IFFT → x_T → DDIM standard (aucun changement de pipeline).
- **Détection** : DDIM inversion `x'_T = D†θ(x'_0)` avec **prompt vide** (prompt inconnu), 50 pas par défaut ; distance L1 sur M : `d = (1/|M|) Σ_{i∈M} |k*_i − F(x'_T)_i|`, détecté si `d < τ`.
- **3 types de clés** : **Zeros** (clé = 0, disque : invariant shifts/crops/dilatations, mais très non gaussien, une seule clé possible, test stat inapplicable car σ≈0) ; **Rand** (clé gaussienne fixe : meilleure qualité, multi-clés, mais **pas invariant** aux rotations) ; **Rings** (valeur constante par anneau, tirée gaussienne : invariant rotation, quasi gaussien, multi-clés). Rings = meilleur compromis.
- **P-value** (Rand/Rings) : `y = F(x'_T)`, H0 : y ~ N(0, σ²I) avec `σ² = (1/|M|) Σ|y_i|²` estimé par image ; score `η = (1/σ²) Σ_{i∈M} |k*_i − y_i|²` ; sous H0 χ² non centré à |M| degrés de liberté, paramètre `λ = (1/σ²)Σ|k*_i|²` ; `p = Φ_χ²(η)` (scipy `ncx2.cdf`). Seuil sur p = FPR contrôlé. Hypothèse gaussienne vérifiée sur images naturelles (µ=−0.005, σ=0.996).
- **Zero-bit** : indique « marqué par cette clé / pas marqué », ne code pas de message. Identification multi-clés testée (Table 5, Bonferroni, FPR 1e-6) : Rand 1.00 jusqu'à 1000 clés, Rings 0.99/0.99/0.98/0.98 pour 50/100/500/1000.

## Implémentation / protocole
- Modèles : Stable Diffusion v2 (guidance 7.5) + ImageNet 256×256 (guided-diffusion). 1 GPU RTX A4000. 1000 images marquées + 1000 non marquées par run, moyenne sur 5 seeds. Baselines : DwtDct, DwtDctSvd (celui de SD), RivaGAN, Stable Signature (SD seulement).
- 6 attaques : rotation 75°, JPEG 25 %, crop+scale 75 %, flou gaussien 8×8, bruit σ=0.1, color jitter brightness 6. Métriques : AUC, TPR@1%FPR (« T@1%F »), FID, CLIP score.

## Résultats clés (Stable Diffusion, AUC / TPR@1%FPR propre → adversarial moyen)
| Méthode | Propre | Adversarial (moy.) | FID (réf. 25.29) |
|---|---|---|---|
| DwtDct | 0.974 / 0.624 | 0.574 / 0.092 | 25.10 |
| DwtDctSvd | 1.000 / 1.000 | 0.702 / 0.262 | 25.01 |
| RivaGAN | 0.999 / 0.999 | 0.854 / 0.448 | 24.51 |
| Tree-Ring Zeros | 0.999 / 0.999 | 0.963 / 0.715 | 26.56 |
| Tree-Ring Rand | 1.000 / 1.000 | 0.918 / 0.702 | 25.47 |
| **Tree-Ring Rings** | 1.000 / 1.000 | **0.975 / 0.694** | 25.93 |
CLIP : Rings 0.364 vs 0.363 de référence. ImageNet : Rings 0.999 → 0.966 AUC (T@1%F 0.603) ; FID 17.68 vs 17.73.
- **AUC par attaque (SD, Table 2)** Rotation / JPEG / Crop&Scale / Flou / Bruit / Jitter : Rings 0.935 / 0.999 / 0.961 / 0.999 / 0.944 / 0.983 ; **Rand : rotation 0.486** ; Zeros : bruit 0.877, jitter 0.907 ; Stable Signature : rotation 0.658, **flou 0.565**, bruit 0.731 ; RivaGAN : **rotation 0.173**.
- Ablations : détection quasi indépendante du nombre de pas génération/détection (AUC ~0.96-0.97 dès 10 pas de détection, 0.808 à 0 pas) ; rayon plus grand = plus robuste mais FID plus haut (rayon 16 jugé bon compromis) ; robuste jusqu'à guidance 18 sans la connaître ; encore détectable avec flou noyau 40 ; Fig. 9 : plusieurs attaques combinées (1 à 6) dégradent progressivement.

## Limites & angles hackathon
- **Requiert DDIM** à la génération (et inversion exacte) ; si le sampler change, à adapter. Précision limitée par l'inversion DDIM (pistes : EDICT, inversion plus précise).
- **Vérifiable uniquement par le propriétaire du modèle** (poids requis pour l'inversion) : pas de détection tierce sans API ; avantage : pas d'attaque white-box ni de vérification d'échec par l'attaquant.
- **Zero-bit** : pas de payload ; capacité multi-clés (identifier un utilisateur parmi beaucoup) pas encore bien cernée (texte : « unclear »).
- **Robustesse géométrique** : Rand fragile à la rotation (0.486) ; Zeros fragile au bruit/jitter ; seul Rings est équilibré mais TPR@1%FPR adversarial reste ~0.69-0.72 (inférence : des attaques combinées/fortes restent un levier).
- Angles défi (inférence) : attaquer l'inversion (régénération par un autre modèle/VAE, edits qui cassent x_T), rotations/crops forts, empilement d'attaques ; côté défense, tester `ncx2.cdf` pour p-value sur nos images.
