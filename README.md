# Incoming PHd — European AI Championship Paris (21-22 nov. 2026)

Tout ce qu'il faut pour démarrer le hackathon : fiches de papiers, code officiel installé, benchmarks, accès GPU.

## Démarrage rapide
```bash
git clone --recursive <url-du-repo>
cd European_AI_Championship_Paris
bash scripts/setup.sh      # venvs uv + poids des modèles (~2,1 Go) + CIFAR-10
```
Sur JURECA, exporter d'abord les variables uv de `infra/jureca_gpu.md` (sinon le `$HOME` déborde).

## Contenu
| Dossier | Quoi |
|---|---|
| `papers/context/` | **Fiches de synthèse à lire d'abord** (index : `papers/README.md`). Watermarking : commencer par `06_organizer_talk_watermarking.md`. |
| `papers/pdf/`, `papers/txt/` | Papiers originaux + texte brut (grep). |
| `repos/` | Code officiel en submodules (videoseal/Pixel Seal, model-equality-testing, SCRUB) + `_smoke/` (tests et benchmark). Lire `repos/README.md`. |
| `repos/_smoke/out/` | Résultats : benchmark de 6 watermarks sur 100 images COCO (`bench_100.log`). |
| `repos/data/coco_val100/` | 100 images COCO val2017 de test. |
| `infra/jureca_gpu.md` | Accès au cluster GPU (JuDoor, MFA, SSH, Slurm, projet `training2666` ; chacun utilise son propre login). |
| `CLAUDE.md` | Consignes pour Claude Code (priorités, où chercher). |

Non versionné (re-téléchargé par `setup.sh`) : `.venv/`, `repos/videoseal/ckpts/`, `repos/data/cifar10_hf/`.
