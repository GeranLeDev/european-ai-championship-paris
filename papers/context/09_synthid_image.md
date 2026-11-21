# CONTEXTE AGENT — SynthID-Image (Google DeepMind, arXiv 2510.09263)
Source vérifiée : texte complet lu. Pas de code ni poids publics (SynthID Detector via synthid.com).

## En une phrase
Papier "retour d'expérience" sur un watermarking invisible **post-hoc, indépendant du modèle générateur** (encodeur/décodeur neuronaux), déployé sur >10 milliards d'images et frames vidéo chez Google ; l'architecture réelle n'est PAS divulguée, seule une variante externe **SynthID-O** (136 bits, 512×512) est évaluée contre 7 baselines.

## Threat model & exigences de déploiement
- **Provenance ≠ détection d'IA** : le watermark prouve l'origine ; les détecteurs "IA ou pas" sont jugés moins fiables. Les métadonnées C2PA sont triviales à supprimer, donc le watermark les complète.
- **Blind watermarking** : le détecteur n'a que l'image marquée (ni original, ni prompt). Le fait que Google marque est public (pas de stéganographie).
- **Exigences** : qualité (invisible), efficacité/robustesse (TPR à FPR bas), payload multi-bit, sécurité, efficacité. Cible d'encodage : surcoût "single-digit percentage" vs temps de génération (objectif, pas de mesure donnée). Le décodage tourne des ordres de grandeur plus souvent que l'encodage → débit de décodage = goulot probable. Aucun chiffre de débit/latence n'est publié.
- **Objectifs attaquant** : suppression (faux négatif), falsification/forgery (faux positif), extraction de modèle, extraction de secret (clés du payload), attaques sur le payload.
- **Capacités** : black-box (API) à white-box ; accès à des paires (original, marqué) = très facilitant pour du post-hoc ; budget de requêtes/calcul.
- **Outils** : exemples adversariaux (universels), modèles substituts (transfert), attaques de **re-génération** (diffusion/VAE).
- **3 modes de déploiement** : service interne (surface minimale), as-a-service (les tiers ont des paires + secrets ; payload distingue les clients), open models (paires partout ; suppression devient triviale par adversarial examples, la forgery gagne en importance, réductible par payload dérivé de clés privées). Encodeur et décodeur se déploient séparément.
- **FPR** : le seul chiffre cible donné est **0,1 % FPR** (évaluation). L'AUROC est jugé trop peu sensible ; les FPR bas ne doivent pas être simulés (hypothèse binomiale trompeuse) mais mesurés sur de gros jeux.
- **Protocole de détection / calibration** : seuil κ unique choisi pour toutes les transformations (évaluer chaque transformation avec son propre seuil = trop optimiste). Décision par **p-valeurs conformes** non paramétriques sur jeu de calibration tenu à part (non marqué pour H0, marqué pour H1) : `ρ_k(x) = (1 + Σ_i 1[(2k−1)f0(x) ≥ (2k−1)f0(x_i^(k))]) / (n+1)` ; seuiller ρ0 à α garantit FPR ≤ α, ρ1 garantit TPR ≥ 1−α ; abstention variable pour contrôler FPR et TPR ensemble. Calibration mise à jour en ligne possible (avec transformations dans le set : garanties dégradées à 2α, jamais observé empiriquement). Payload : p-valeur par bit + tests multiples (Holm/Hochberg/Simes) ; plusieurs payloads acceptés = signe de manipulation (abstention).

## Méthode (ce qui est divulgué)
- **Encodeur** `f(x, c)` ajoute la marque ; **décodeur** `g` renvoie un logit de détection `g0` + un logit par bit du payload `c ∈ {±1}^C`. Détection et payload sont entraînés **séparément** (Éq. 2 et 3) ; le payload n'est pas nécessaire pour détecter.
- Objectif : min E[ℓ(g(τ(f(x))),+1) + ℓ(g(τ(x)),−1)] s.c. `d(x, f(x)) ≤ ε` (d = métrique perceptuelle), τ ~ distribution de transformations préservant la sémantique. Cross-entropy en pratique, seuil calibré post-entraînement ; hypothèse x marqué/non marqué de même distribution corrigée à la calibration.
- Contrôle des compromis via mix de pertes, augmentations et encodage du payload ; **cascade de décodeurs** (rapide peu robuste d'abord) mentionnée comme levier.
- **Défenses de sécurité** : entraînement robuste (augmentations ; adversarial training du décodeur possible) ; robustesse testée contre re-génération faible (VAE off-the-shelf) ; **aléatoire** (même seed+prompt → même sortie, contre collusion ; mais diversité des marques pour contenu prévisible) ; **spécificité** (marque dépendante du contenu, pertes anti-échange de watermark) ; **filtrage** (ne pas marquer images quasi uniformes). Stratégie : défendre contre plus fort que le réel (white-box → couvre black-box). Objectif réel : rendre le black-box infaisable à l'échelle (système propriétaire).
- **Non divulgué** : architecture, taille, données et durée d'entraînement, pertes exactes, clé/secret, format du payload, résolution interne de SynthID-Image (résolution fixée par contraintes d'efficacité, redimensionnement sans artefacts).
- **Fingerprinting** (embeddings robustes en base interne + recherche de similarité) combiné au watermark : anti-forgery (il faut le bon payload ET un embedding stocké) ; coût de stockage, FP qui croissent avec la base → règles de rétention.

## Évaluation (SynthID-O uniquement)
- **Baselines** (checkpoints officiels, paramètres par défaut, sans correction d'erreur) : StegaStamp 100 b/400², TrustMark-P et -Q 100 b/256², InvisMark 100 b/256², WAM 32 b/256², VideoSeal-0.0 96 b/256², VideoSeal-1.0 256 b/256².
- **Qualité (humains)** : 10 000 notes/modèle (5 000 marqué, 5 000 non marqué) sur 1 000 images Imagen 2 en 1536×1536, 25 catégories ; 10 images pièges, raters <70 % exclus ; métrique = hausse (points de %) du taux d'artefacts identifié (borne basse). Classement : SynthID-O meilleur, puis VideoSeal-1.0, InvisMark, WAM ; StegaStamp pire. VideoSeal-0.0 et TrustMark-Q ont de bons PSNR/SSIM/LPIPS mais mauvais résultats humains → métriques auto peu corrélées. En comparaison côte à côte, tous sauf SynthID sont repérés (note 16). Études internes : cible 50 % ± 10 % de chance.
- **Robustesse** : 10 000 images ImageNet val, 512×512, 30 transformations en 6 catégories (Color, Combination, Noise, Overlay, Quality, Spatial) ; modes Random et **Worst** (force max) ; seuil calibré à **0,1 % FPR** en moyenne sur les transformations worst.

| TPR @0,1 % FPR (512²) | SynthID-O | meilleur autre |
|---|---|---|
| Aggregated | 99,98 % | WAM 90,62 % (+9,36 pts) |
| Aggregated Worst | 99,72 % | WAM 83,37 % (+16,35 pts) |
| Combination Worst | 98,06 % | WAM 55,96 % |
| Spatial Worst | 99,97 % | WAM 76,04 % |

  Aucun modèle ne gagne partout (StegaStamp domine certaines catégories au prix de la qualité). Plus faibles en worst : InvisMark 60,11 % (agg.), TrustMark-P 54,15 %, Spatial Worst StegaStamp 15,25 %. SynthID-O seul >99 % en worst agrégé.
- **Détection via payload** (Table 2) : SynthID-O Aggregated Worst [98,84 %, 98,90 %], Combination Worst ≈ 92 % ; WAM [69,11 %, 69,91 %].
- **Bit accuracy (512²)** : SynthID-O (136 b) Aggregated 99,59 %, Worst 98,07 %, Combination Worst 89,46 % ; VideoSeal-1.0 (256 b) 89,60 / 83,53 % ; WAM (32 b) 90,62 / 85,03 % ; non directement comparables (tailles différentes).
- **Résolution native ImageNet (moy. 469×387)** : agrégé 98,72 %, worst 97,22 % (+10,64 / +17,17 pts vs meilleur autre).
- **Résolution propre à chaque modèle** : SynthID-O 99,98 % / 99,72 % ; meilleurs autres ≈ 86,56 % (VideoSeal-1.0) et 79,38 % (WAM, worst).
- Aucune évaluation d'attaque adversariale ou de re-génération n'est chiffrée dans le papier (seulement affirmée comme testée).

## Leçons de déploiement
1. Post-hoc = un seul schéma pour tous les modèles (payload pour distinguer), plus simple à débugger/activer ; coût = perte de qualité inhérente et surcoût de calcul.
2. La qualité se juge par humains sur contenu IA réel et cas limites (N&B, logos, dégradés, pixel art) ; PSNR/SSIM seulement en relatif.
3. Un seul seuil de déploiement ; évaluer à FPR très bas sur vraies données.
4. **Versioning** : l'encodeur fixe la version (contenu déjà en circulation) ; le décodeur est mis à jour en continu. Les menaces traitables côté encodeur (diversité, anti-extraction) sont prioritaires car irréparables après déploiement ; les vulnérabilités peuvent s'hériter entre versions.
5. Open models : la suppression est triviale, miser sur anti-forgery (payload clé privée, redondance).
6. Combiner watermark + C2PA + fingerprinting.

## Limites / angles hackathon
- Reproduction impossible : ni code ni poids ; SynthID-O dispo seulement via partenariat. Pour expérimenter, utiliser Pixel Seal/VideoSeal (voir `01_pixel_seal.md`) comme proxy.
- Évaluation faite par les auteurs sur leur propre modèle ; baselines à paramètres par défaut ; pas de budget adversarial chiffré ni re-génération diffusion évaluée → angle d'attaque pertinent (inférence : tester SynthID-like via regen diffusion, adversarial, collusion/moyenne d'images marquées).
- Bonnes pratiques réutilisables en 24h : seuil unique calibré à 0,1 % FPR, mode "worst", p-valeurs conformes avec abstention, catégories de transformations, qualité humaine plutôt que PSNR.
- Limites avouées : le watermark seul ne résout pas la désinformation ; marques fractionnelles, payloads plus grands, signatures cryptographiques (détection publique), sécurité white-box des modèles ouverts et versioning restent ouverts.
