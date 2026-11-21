#!/usr/bin/env bash
# One-shot setup: submodules, uv venvs, checkpoints and data.
# Usage: bash scripts/setup.sh            (from the repo root, Mac or JURECA)
#        SKIP_VENV=1 bash scripts/setup.sh (only download weights/data)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

git submodule update --init --recursive

dl() { [ -s "$2" ] || { echo "-> $2"; curl -fL --retry 3 -o "$2" "$1"; }; }

# ---------- venvs ----------
if [ -z "${SKIP_VENV:-}" ]; then
  ( cd repos/videoseal
    [ -d .venv ] || uv venv -q -p 3.10 .venv
    export VIRTUAL_ENV=.venv
    uv pip install -q torch torchvision torchaudio
    # decord has no macOS ARM wheel: fall back to installing without it
    uv pip install -q -r requirements.txt || grep -v -i decord requirements.txt | uv pip install -q -r /dev/stdin
    uv pip install -q --no-deps -e . && uv pip install -q requests scipy )
  ( cd repos/model-equality-testing
    [ -d .venv ] || uv venv -q -p 3.11 .venv
    VIRTUAL_ENV=.venv uv pip install -q -e ./model_equality_testing ipykernel tqdm scipy pandas transformers gdown )
  ( cd repos/SCRUB
    [ -d .venv ] || uv venv -q -p 3.10 .venv
    VIRTUAL_ENV=.venv uv pip install -q torch torchvision numpy scipy matplotlib pandas scikit-learn tqdm ipython ipykernel wandb pyarrow )
fi

# ---------- watermark checkpoints (~2.1 GB) ----------
CK=repos/videoseal/ckpts; mkdir -p "$CK"
dl https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth "$CK/pixelseal_checkpoint.pth"
dl https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth "$CK/videoseal_y_256b_img.pth"
HF=https://huggingface.co/tangtianzhong/img-wm-torchscript/resolve/main
for f in hidden_encoder_48b hidden_decoder_48b trustmark_encoder_q trustmark_decoder_q \
         wam_encoder wam_decoder mbrs_256_m256_encoder mbrs_256_m256_decoder \
         cin_nsm_encoder cin_nsm_decoder; do
  dl "$HF/$f.pt" "$CK/$f.pt"
done

# ---------- data ----------
mkdir -p repos/data/cifar10_hf
for s in train test; do
  dl "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/$s-00000-of-00001.parquet" \
     "repos/data/cifar10_hf/$s.parquet"
done
# repos/data/coco_val100/ is committed (15 MB).

echo "Setup OK. Smoke tests: see repos/README.md"
