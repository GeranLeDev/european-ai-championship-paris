# CONTEXTE AGENT — Pixel Seal (Meta FAIR, arXiv 2512.16874, déc. 2025)
Source vérifiée : texte complet lu. Code + poids : https://github.com/facebookresearch/videoseal · site : https://facebookresearch.github.io/meta-seal

## En une phrase
Watermarking invisible **post-hoc** (après génération) multi-bit pour images ET vidéos : un embedder ajoute un message de 256 bits, un extractor le relit même après édition ; SOTA robustesse + imperceptibilité.

## Pipeline (formules à connaître)
- Embedding : `xw = x + α · (w ⊙ JND(x))`, `w = Emb(x, m)`, `m ∈ {0,1}^256`, `w ∈ [-1,1]^{H×W×3}`.
- Extraction : `m̃ = Ext(x̃w) ∈ [0,1]^nbits` → seuil 0.5 → `m̂`.
- **Détection oui/non avec garantie statistique** : on embarque toujours le même `m`, puis
  `p-value = Σ_{k ≤ dH(m, m̂)} C(nbits, k) / 2^nbits` (proba qu'une image non marquée matche autant de bits par hasard). Métrique rapportée : `−log10 p` (permet de comparer des méthodes avec des tailles de message différentes). Pas besoin de calibrer sur des images non marquées.
- Haute résolution (inférence) : image redimensionnée en 256×256 → `w` calculé en basse rés → upsamplé à la résolution d'origine → multiplié par la carte JND calculée **en pleine résolution**.
- Entraînement sim. inférence : `m̃ = Ext(resize↓(τ(xw)))`, `xw = x + α·(resize↑(w) ⊙ JND(x))`, `w = Emb(resize↓(x), m)` ; τ = augmentation aléatoire différentiable.
- Augmentations non différentiables (JPEG…) : straight-through estimator `x̃w = xw + nograd(T(xw) − xw)`.

## 3 contributions clés
1. **Adversarial-only loss** : AUCUNE perte perceptuelle (pas de MSE/LPIPS). Seulement `Lmsg` (BCE sur les bits) + `Ladv` (discriminateur patch-based de Rombach et al. 2022, dual-hinge). λmsg = 1.0, λadv = 0.1.
   - **Watermark boosting** : le discriminateur voit `x̂w = x + β(xw − x)`. β>1 → plus invisible, moins robuste (β=2.5 : PSNR 50.1, combined 0.86) ; β<1 → plus robuste, plus visible (β=0.5). Défaut β=1.
2. **Schedule en 3 étapes** (sinon la bit accuracy reste à 50 % = échec total) :
   - Étape 1 : seulement Lmsg, α = α0 = 1.0 (watermark très visible) jusqu'à saturation de la bit accuracy.
   - Étape 2 : ajout de Ladv + α décroît de 1.0 vers α1 = 0.2 en cosinus : `α(t) = α1 + (α0−α1)·cos(π/2·φ)`, `φ = clip((t−Nstart)/(Nend−Nstart), 0, 1)`.
   - Étape 3 : finetune à α = 0.2 (données/attaques spécifiques possibles).
3. **High-res adaptation** : JND + discriminateur + attaques appliqués en haute résolution pendant l'entraînement ; tailles aléatoires dans [256, 768] (aspect ratios variés). Gain fort sur géométrique (0.81→0.93) et combiné (0.68→0.94).
4. **Vidéo — temporal watermark pooling** (inférence seulement, sans finetune vidéo) : average pooling temporel (kernel = stride = k) après le d-ième bloc descendant du U-Net + un-pooling (repeat) symétrique. k=4, d=1 → ×2.2 de vitesse sans perte. (En éval : k=4, d=2.)

## Implémentation
- Embedder : U-Net 43.8M params. Extractor : ConvNeXt-v2 Tiny 33.4M params. 256 bits.
- SA-1B, 600k steps, batch 256, AdamW lr 5e-4, cosine decay, warm-up 20k steps.

## Résultats clés (bit acc / −log10 p)
| Images Meta AI | Identity | Valuemetric | Compression | Géométrique | Combiné* |
|---|---|---|---|---|---|
| Pixel Seal | 1.00 / 75.8 | 0.97 / 68.1 | 0.98 / 68.9 | 0.95 / 63.9 | 0.91 / 50.3 |
| Video Seal 1.0 | 0.98 / 69.4 | 0.95 / 60.7 | 0.95 / 60.6 | 0.83 / 42.9 | 0.69 / 16.0 |
| InvisMark (94 bits) | 1.00 / 28.3 | 0.89 / 22.0 | 0.98 / 26.1 | 0.77 / 14.8 | 0.94 / 22.1 |
*Combiné = JPEG + crop + luminosité (ex. JPEG40, crop 0.71, bright 0.5).
- Imperceptibilité (Meta AI) : PSNR 48.9, SSIM 0.9905, LPIPS 0.0013, JND 0.34 (1 JND = 75 % des observateurs voient la différence).
- **Points faibles mesurés (= angles d'attaque)** : crop 0.32 → 0.58 bit acc ; brightness 0.1 → 0.80 ; perspective 0.8 → 0.87 ; vidéo H.264 CRF 40 → 0.77, CRF 50 → 0.52, H.265 CRF 50 → 0.54 ; combiné vidéo (H264 30 + crop + bright) → 0.76, (H264 40) → 0.51.
- Très robuste : flips, rotations (≤90°), blur, hue, grayscale, JPEG ≥ 40 (≥ 0.96).

## Observations locales (notre smoke test, 1 seule image 1904×1080 — à confirmer sur plus d'images)
Script : `repos/_smoke/pixelseal_smoke.py` (lancer depuis `repos/videoseal`). PSNR 52.0 dB, 256 bits.
- identity 1.000 · JPEG40 0.969 · crop 0.32 aire 0.992 · brightness 0.5 1.000 · image non marquée 0.516 (≈ hasard, OK).
- **L'ordre des attaques compte énormément** : crop0.71 → brightness0.5 → JPEG40 = **0.531 (watermark détruit)** ; JPEG40 → crop → brightness = 0.898 (≈ papier 0.86-0.91) ; brightness0.5 → JPEG40 = 0.836 ; brightness0.5 → JPEG60 = 0.898 ; brightness0.5 → JPEG80 = 0.965.
- ⇒ Piste d'attaque bon marché : **assombrir puis compresser** (la compression sur une image sombre écrase le signal). Le papier ne précise pas l'ordre du combiné.

## Limites / angles hackathon
- Pas de robustesse évaluée contre la purification par diffusion, la régénération ou les attaques adversariales ciblées sur l'extractor (inférence : vulnérabilités probables, non testées dans le papier).
- Ablations : perte MSE avec λ=1.0 → le modèle n'apprend rien ; sans JND → PSNR 43.1, JND 1.34 (visible).
- Idées : (attaque) retirer/forger un watermark avec un budget de qualité (PSNR/LPIPS), régénération diffusion, crop fort + compression ; (défense) détecteur de provenance, outil de vérification `−log10 p` ; (produit) pipeline de watermarking vidéo rapide via pooling.

## Famille encodeur-décodeur (slide organisateurs) et baselines disponibles
Recette commune : un embedder et un extractor, avec des édits aléatoires entre les deux. Les méthodes diffèrent par la résolution, l'imperceptibilité, la stabilité d'entraînement et la localisation.
- **HiDDeN** : premier embedder-extractor profond, entraîné de bout en bout avec édits au milieu (48 bits).
- **TrustMark** : résolution de travail fixe puis resize → toute taille d'image (Pixel Seal reprend cette idée en haute résolution).
- **SynthID-Image** (DeepMind) : watermarking post-hoc à l'échelle d'internet. Pas de code public. Pixel Seal reprend ses prompts pour générer les 1000 images Meta AI d'évaluation.
- **Video Seal** : même recette appliquée à la vidéo, frame par frame (0.0 : 96 bits, 1.0 : 256 bits).
- **Pixel Seal** : entraînement adversarial seul, modulation JND (voir plus haut).
- **Watermark Anything (WAM)** : messages localisés, répond à « où l'image est-elle marquée ? » (32 bits, détection de forgeries/splicing).
- **Dans le repo `videoseal`** : HiDDeN, MBRS, CIN, WAM et TrustMark sont utilisables comme baselines (`videoseal.load("baseline/<method>")`, `videoseal/models/baselines.py`). Il faut d'abord télécharger leurs checkpoints TorchScript dans `ckpts/` (`hidden_encoder_48b.pt`, etc.), voir `repos/videoseal/docs/baselines.md`. Ils ne sont **pas encore téléchargés**. Video Seal 0.0/1.0 et Pixel Seal ont déjà leur config (`videoseal/cards/`).
