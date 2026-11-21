# Papiers de préparation — European AI Championship Paris (21-22 nov. 2026)

Papiers recommandés par les organisateurs (CISPA + Meta). Thème commun : **sécurité / confiance des modèles d'IA**.

| # | Papier | Thème | Notes détaillées | Code |
|---|--------|-------|------------------|------|
| 1 | Pixel Seal (Meta, arXiv 2512.16874) | Watermarking invisible image/vidéo | [notes/01_pixel_seal.md](notes/01_pixel_seal.md) | https://github.com/facebookresearch/videoseal |
| 2 | Model Equality Testing (ICLR 2025, arXiv 2410.20247) | Audit boîte noire d'API LLM (MMD) | [notes/02_model_equality_testing.md](notes/02_model_equality_testing.md) | https://github.com/i-gao/model-equality-testing |
| 3 | Towards Unbounded Machine Unlearning / SCRUB (NeurIPS 2023, arXiv 2302.09880) | Machine unlearning + MIA | [notes/03_unbounded_unlearning_scrub.md](notes/03_unbounded_unlearning_scrub.md) | https://github.com/Meghdad92/SCRUB |

## Watermarking — talk organisateur + papiers cités (ajoutés le 21/11, 14h)
Le talk de l'organisateur Pierre Fernandez (Meta), **« Invisible image watermarking between research and production »**, est résumé dans [context/06_organizer_talk_watermarking.md](context/06_organizer_talk_watermarking.md). **À lire en premier** pour tout challenge watermarking.

| Fiche | Papier | Code / poids |
|---|---|---|
| [05_hidden](context/05_hidden.md) | HiDDeN (ECCV 2018, 1807.09937) — la recette | `repos/videoseal` baseline `hidden` (48 bits) |
| [07_watermark_anything](context/07_watermark_anything.md) | WAM (ICLR 2025, 2411.07231) — localisation | baseline `wam` + github facebookresearch/watermark-anything |
| [08_video_seal](context/08_video_seal.md) | Video Seal (2412.09492) | `videoseal.load("videoseal")` |
| [09_synthid_image](context/09_synthid_image.md) | SynthID-Image (DeepMind, 2510.09263) | aucun code public |
| [10_trustmark](context/10_trustmark.md) | TrustMark (Adobe, 2311.18297) | baseline `trustmark` (Q) |
| [11_editguard](context/11_editguard.md) | EditGuard (CVPR 2024, 2312.08883) — baseline localisée | non installé |
| [12_stable_signature](context/12_stable_signature.md) | Stable Signature (ICCV 2023, 2303.15435) | non installé |
| [13_tree_ring](context/13_tree_ring.md) | Tree-Ring (NeurIPS 2023) | non installé |
| [14_distseal](context/14_distseal.md) | DistSeal (2601.16140) | non installé |

PDF dans `pdf/`, texte brut dans `txt/`. Le texte et les notes du speaker du deck sont dans `txt/organizer_slides_raw.txt`.

> **Focus : ces 3 papiers.** `context/04_complementary_papers.md` = bonus pour combler leurs points faibles, pas une base de travail.

## Arborescence
- `context/` — **synthèses compactes vérifiées sur le texte complet (~1-1,5k mots), à donner en priorité aux agents** (formules, hyperparams, chiffres, pseudo-code, angles hackathon)
- `pdf/` — PDFs originaux
- `txt/` — texte brut extrait (grep-able, ~9k / 25k / 18k mots)
- `notes/` — analyses longues (~5k mots, générées par haiku, non vérifiées ligne à ligne) : seulement si `context/` ne suffit pas

## Lien commun (inférence)
Les trois touchent à la **vérification de propriétés d'un modèle ou de ses sorties** :
provenance du contenu (watermark), identité du modèle servi (equality testing), preuve d'oubli (unlearning + MIA).
Challenges probables : attaque vs défense sur l'un des trois, ou un outil d'audit qui combine ces idées.
