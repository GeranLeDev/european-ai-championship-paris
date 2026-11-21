# Repos officiels des 3 papiers — installés et testés (Mac M-series, MPS)

Chaque repo a son propre venv `uv` dans `<repo>/.venv`. Smoke tests dans `_smoke/`.

| Repo | Papier | Python | Lancer le smoke test |
|---|---|---|---|
| `videoseal/` | Pixel Seal | 3.10 | `cd videoseal && .venv/bin/python ../_smoke/pixelseal_smoke.py` |
| `model-equality-testing/` | Model Equality Testing | 3.11 | `model-equality-testing/.venv/bin/python _smoke/met_smoke.py` |
| `SCRUB/` | SCRUB unlearning | 3.10 | `cd SCRUB && .venv/bin/python ../_smoke/scrub_smoke.py` |

## Pièges rencontrés (déjà corrigés)
- **videoseal** : `decord` n'a pas de wheel macOS ARM → package installé avec `--no-deps` (decord seulement utile pour l'entraînement vidéo). `requests` manquait. **Toujours lancer depuis `videoseal/`** (chemins relatifs `configs/` et `ckpts/`). Poids Pixel Seal : `videoseal/ckpts/pixelseal_checkpoint.pth`. API : `videoseal.load("pixelseal")`, `model.embed(img)` → `imgs_w`, `msgs` ; `model.detect(x)["preds"][:, 1:] > 0` → bits.
- **model-equality-testing** : dépendances non déclarées `tqdm`, `scipy`, `pandas`, `transformers` (+ `gdown` pour le dataset 37 GB, non téléchargé). Les complétions doivent être ≤ L caractères avant `pad_to_length`. API : `run_two_sample_test(s1, s2, pvalue_type="permutation_pvalue", stat_type="mmd_hamming", b=...)`.
- **SCRUB** : code en notebooks (`large_scale_unlearning.ipynb`, `MIA_experiments.ipynb`), dépend de `wandb`. Le smoke test réimplémente la boucle SCRUB en ~30 lignes avec le `ResNet18` du repo. CIFAR-10 téléchargé dans `repos/data/`.

## Résultats des smoke tests (21/11/2026, Mac M-series MPS)
- **Pixel Seal** : PSNR 52.0 dB ; bit acc identity 1.000, JPEG40 0.969, crop 0.32 0.992, image non marquée 0.516. L'ordre crop→assombrir→JPEG40 détruit le watermark (0.531) — voir `papers/context/01_pixel_seal.md`.
- **Model Equality Testing** : P vs P p=0.085 (non rejeté), P vs décalé p=0.000 (rejeté).
- **SCRUB** (classe 1 de CIFAR-10, sous-ensemble de 20k, ResNet18 du repo) : forget 0.974→0.000, retain 0.900→0.996, test autres classes 0.680→0.770, MIA basique 0.658→0.985 (sur-oubli détectable → motive SCRUB+R).
  - CIFAR-10 chargé depuis le miroir parquet HF (`data/cifar10_hf/`) car le serveur de Toronto est trop lent. Modèle original en cache : `data/scrub_smoke_orig.pt`.
  - Réglages qui comptent : **forget batch 512** (32 → la montée de KL diverge vers NaN), lr 5e-4, kd_T 4, alpha 0.001, gamma 0.99 ; msteps 3 (2 dans le repo → oubli partiel, 0.333).

## Comparatif des baselines de watermark (21/11, 14h, Mac CPU, `_smoke/baselines_smoke.py`)
- Lancement : `cd videoseal && .venv/bin/python ../_smoke/baselines_smoke.py`.
- Les baselines se chargent via `setup_model_from_checkpoint("baseline/<m>")`, et non via `videoseal.load`. Les poids TorchScript sont dans `videoseal/ckpts/` (HF `tangtianzhong/img-wm-torchscript`).
- Mesure : bit accuracy sur **1 seule image** (`assets/imgs/1.jpg`, 512 px). C'est donc indicatif ; à refaire sur N images avant de citer un chiffre.

| Modèle | bits | PSNR | identity | JPEG40 | crop 32 % | lum. 0,5 | crop→sombre→JPEG40 | non marquée |
|---|---|---|---|---|---|---|---|---|
| HiDDeN | 48 | 30,4 | 1,000 | 0,979 | 0,521 | 0,917 | 0,479 | 0,500 |
| TrustMark-Q | 100 | 44,2 | 1,000 | 1,000 | 0,410 | 1,000 | 0,490 | 0,580 |
| WAM | 32 | 42,6 | 1,000 | 0,938 | 1,000 | 1,000 | 0,562 | 0,344 |
| MBRS | 256 | 48,0 | 0,988 | 0,949 | 0,508 | 0,977 | 0,539 | 0,488 |
| Video Seal 1.0 | 256 | 49,5 | 1,000 | 0,867 | 0,824 | 1,000 | 0,559 | 0,516 |
| Pixel Seal | 256 | 50,7 | 1,000 | 0,848 | 1,000 | 1,000 | 0,500 | 0,516 |

- **CIN** échoue sur Mac : le TorchScript a été tracé sur CUDA. À tester sur JURECA.
- **L'attaque combinée crop→assombrir→JPEG40 casse les 6 méthodes** (≈ 0,5, autant que le hasard).
- Crop seul : seuls WAM et Pixel Seal tiennent ; HiDDeN, TrustMark et MBRS tombent à ≈ 0,5.
- Les écarts des images non marquées (0,34-0,58) viennent des petits messages (32 bits) sur une seule image. C'est pour ça qu'il faut un **test binomial / p-value**, pas un seuil brut sur la bit accuracy.

## Comparatif sur 100 images COCO val2017 (21/11, `_smoke/baselines_bench.py`, résultats dans `_smoke/out/bench_100.{log,json}`)
- Images : `data/coco_val100/`, résolution native (~640×480), CPU.
- Mesures :
  - **bit acc** = moyenne sur 100 images ;
  - **TPR** = part des images marquées détectées par p-value binomiale < 1e-3 ;
  - **FPR** = fausses alertes sur les 100 images non marquées : **0 % pour toutes les méthodes**.
- Durée totale : ~16 min en CPU. Ce tableau remplace celui sur 1 image ci-dessus.

| Méthode | bits | PSNR | JPEG40 | crop 32 % | lum. 0,5 | rot 10° | flou σ2 | resize ½ | crop→sombre→JPEG40 |
|---|---|---|---|---|---|---|---|---|---|
| HiDDeN | 48 | 30,4 | 0,98 / 100 % | 0,59 / 2 % | 0,95 / 99 % | 0,83 / 95 % | 0,80 / 76 % | 0,95 / 100 % | 0,52 / 0 % |
| TrustMark-Q | 100 | 42,9 | 0,99 / 100 % | 0,50 / 0 % | 1,00 / 100 % | 0,51 / 0 % | 1,00 / 100 % | 1,00 / 100 % | 0,50 / 0 % |
| WAM | 32 | 38,9 | 0,97 / 98 % | 0,98 / 99 % | 1,00 / 100 % | 0,77 / 40 % | 1,00 / 100 % | 1,00 / 100 % | 0,53 / 0 % |
| MBRS | 256 | 46,1 | 0,96 / 100 % | 0,50 / 0 % | 0,96 / 100 % | 0,50 / 2 % | 0,88 / 100 % | 0,97 / 100 % | 0,50 / 0 % |
| Video Seal 1.0 | 256 | 45,2 | 0,97 / 100 % | 0,52 / 6 % | 1,00 / 100 % | 0,99 / 100 % | 1,00 / 100 % | 1,00 / 100 % | 0,50 / 0 % |
| **Pixel Seal** | 256 | 46,1 | 0,95 / 100 % | **0,95 / 100 %** | 1,00 / 100 % | **0,99 / 100 %** | 1,00 / 100 % | 1,00 / 100 % | 0,51 / **4 %** |

- **Pixel Seal est le seul à tenir toutes les attaques simples** : 100 % de détection partout.
- **L'attaque combinée crop 32 % → luminosité 0,5 → JPEG40 casse les 6 méthodes** (0-4 % de détection), alors que chaque attaque seule est survécue par Pixel Seal.
- Crop seul : seuls WAM et Pixel Seal tiennent. Rotation 10° : seuls Video Seal, Pixel Seal et, en partie, HiDDeN tiennent.
- Le PSNR de Pixel Seal sur COCO (46,1 dB) est plus bas que sur `1.jpg` (50,7 dB) : il dépend de l'image.
