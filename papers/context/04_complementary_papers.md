# CONTEXTE AGENT — BONUS : papiers complémentaires
> ⚠️ SECONDAIRE. Le focus du projet = les 3 papiers initiaux (01_pixel_seal, 02_model_equality_testing, 03_scrub_unlearning). Ce fichier sert seulement à compléter un point faible précis de l'un d'eux ; ne jamais construire le MVP autour de ces papiers.
Source : recherche web du 21/11/2026, infos tirées des abstracts / pages de résumé (PAS du texte complet). Vérifier les chiffres avant de les citer. ★ = priorité pour le hackathon.

## A. Pixel Seal — ce qu'il n'évalue pas : régénération, attaques adaptatives, forgery, édition locale
| Papier | Ce qu'il apporte | Lien / code |
|---|---|---|
| ★ **Invisible Image Watermarks Are Provably Removable Using Generative AI** (Zhao et al., NeurIPS 2024) | Attaque par régénération : bruit dans le latent + reconstruction (VAE/diffusion). Prouvée efficace contre tout watermark à perturbation L2 bornée → Pixel Seal (perturbation pixel, α=0.2) est a priori dans cette classe (inférence). 98 % des RivaGAN retirés à PSNR > 30. | arXiv 2306.01953 · github.com/XuandongZhao/WatermarkAttacker |
| ★ **WAVES: Benchmarking the Robustness of Image Watermarks** (An et al., ICML 2024) | Benchmark standard : distorsions, régénération, attaques adversariales ; métrique TPR @ 0.1 % FPR + dégradation de qualité nécessaire. Cadre d'évaluation prêt à l'emploi pour un MVP attaque/défense. | arXiv 2401.08573 · wavesbench.github.io |
| ★ **UnMarker: A Universal Attack on Defensive Image Watermarking** (Kassis & Hengartner, IEEE S&P 2025) | Attaque sans feedback du détecteur ni connaissance du schéma : 2 optimisations adversariales qui perturbent le spectre de l'image. Rapporte des succès contre SynthID et Stable Signature (chiffres SynthID contestés par Google). Marche mieux avec un léger crop → recoupe la faiblesse crop de Pixel Seal. | arXiv 2405.08363 · github.com/andrekassis/ai-watermark |
| **Leveraging Optimization for Adaptive Attacks on Image Watermarks** (Lukas et al., ICLR 2024) | Attaques adaptatives via des clés de substitution différentiables (Adversarial Noising, Adversarial Compression) ; < 1 h GPU pour faire tomber la détection à ≤ 6.3 % sur 5 schémas. | arXiv 2309.16952 |
| **Diffusion-Based Image Editing for Breaking Robust Watermarks** (2025) | Attaque par édition diffusion, variante guidée par le décodeur du watermark. | arXiv 2510.05978 |
| ★ **WMCopier** (NeurIPS 2025) | *Forgery* no-box : copier un watermark sur une image propre via inversion diffusion superficielle ; succès rapporté contre le détecteur closed-source d'Amazon. Pixel Seal ne traite pas la forgery. | arXiv 2503.22330 |
| **Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models** (NeurIPS 2025) | Forgery avec UNE seule image marquée, sans connaître le schéma ; note que beaucoup de décodeurs ne vérifient pas le lien avec l'image source. | arXiv 2510.20468 |
| **Watermark Anything with Localized Messages (WAM)** (Sander et al., Meta, ICLR 2025) | Watermarking comme segmentation : localise les zones marquées, plusieurs messages de 32 bits dans de petites régions (≤ 10 % de l'image) → réponse aux crops forts / splicing / inpainting. Même équipe que Pixel Seal (baseline dans le papier). | arXiv 2411.07231 · github.com/facebookresearch/watermark-anything (archivé, MIT) |

## B. Model Equality Testing — limites : coût en requêtes, substitutions subtiles, adversaires, logprobs
| Papier | Ce qu'il apporte | Lien / code |
|---|---|---|
| ★ **Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs** (Cai, Shi, Zhao, Song — Berkeley, NeurIPS 2025) | Suite directe : fournisseur **adversarial** (substitution aléatoire, évasion des benchmarks). Les tests statistiques sur texte coûtent cher en requêtes et ratent les substitutions subtiles ; les logprobs sont cassées par le non-déterminisme de l'inférence → propose une vérification matérielle (TEE). | arXiv 2504.04715 · github.com/sunblaze-ucb/llm-api-audit |
| ★ **LLMmap: Fingerprinting for Large Language Models** (Pasquini et al., USENIX Security 2025) | Fingerprinting **actif** : requêtes choisies → identifie le modèle parmi 42 avec > 95 % de précision en ~8 interactions (vs ~10 échantillons/prompt × 25 prompts en MET). Complément idéal : identification (LLMmap) + test d'égalité (MET). | arXiv 2407.15847 · github.com/pasquini-dario/LLMmap |
| **Idiosyncrasies in Large Language Models** (Sun et al., ICML 2025) | Un classifieur (embeddings finetunés) identifie la famille d'un LLM depuis son texte avec 97.1 % de précision (5 modèles) ; le signal survit à la réécriture et à la traduction. Noyau/feature plus fort que Hamming pour le cas inter-familles. | arXiv 2502.12150 · github.com/locuslab/llm-idiosyncrasies |

## C. SCRUB / unlearning — limites : MIA trop faibles, vision seulement, oubli superficiel
| Papier | Ce qu'il apporte | Lien / code |
|---|---|---|
| ★ **Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy** (Hayes, Shumailov, Triantafillou, Khalifa, Papernot — DeepMind, 2024) | Même groupe que SCRUB. Les MIA « population » habituelles (type Basic MIA de SCRUB) surestiment la confidentialité ; les MIA **par exemple** (style LiRA) sont bien plus fortes. L'unlearning inexact peut aussi *augmenter* la fuite sur le retain set. | arXiv 2403.01218 |
| ★ **Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning** (Hu, Fu, Wu, Smith — ICLR 2025) | Relearning attack : un petit finetune sur des données vaguement liées restaure le savoir « oublié » → l'unlearning approximatif ne fait souvent que *masquer*. Testé sur WMDP, TOFU, Harry Potter. | arXiv 2406.13356 · github.com/s-huu/jog_llm_memory |
| **TOFU: A Task of Fictitious Unlearning for LLMs** (Maini et al., 2024) | Benchmark LLM de référence : 200 auteurs fictifs × 20 QA, oublier 2 / 10 / 20 auteurs. Permet de porter SCRUB aux LLM (limite explicite du papier SCRUB). | arXiv 2401.06121 |
| **MUSE: Machine Unlearning Six-Way Evaluation for Language Models** (Shi et al., ICLR 2025) | 6 critères (mémorisation verbatim, mémorisation des connaissances, fuite de confidentialité, utilité, passage à l'échelle, requêtes successives) ; 8 algorithmes sur des modèles 7B (Harry Potter, news). Un seul évite une fuite de confidentialité sévère. | arXiv 2407.06460 · github.com/swj0419/muse_bench |

## Combinaisons intéressantes pour un MVP (inférence)
1. **Banc d'attaque watermark** : Pixel Seal (open source) × régénération (Zhao) + UnMarker + crop/H.264 → courbes qualité vs `−log10 p` façon WAVES ; défense : WAM pour la localisation.
2. **Auditeur d'API LLM** : LLMmap (quel modèle ?) → MET Hamming MMD (est-ce le modèle exact / quantifié ?) → avertissement sur les limites adversariales (Cai et al.).
3. **Vérificateur d'unlearning** : SCRUB(+R) → évaluer avec une MIA par exemple (Hayes) + relearning attack (Hu) + test MET « désappris ≈ retrain ? ».
