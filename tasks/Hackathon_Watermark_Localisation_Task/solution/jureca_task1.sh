#!/bin/bash
# Task 1 (watermark localisation) on JURECA: features -> synthetic data -> U-Net training -> submission.pt
# Setup once (login node):  cd $PROJECT/$USER/Hackathon_Watermark_Localisation_Task
#   uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements.txt scikit-learn
#   .venv/bin/python download_models.py
#   data/ (train, test) + extra/val2017 (COCO val2017) must be present (rsync from the laptop or re-download)
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=06:00:00
#SBATCH --output=output/%j.out
set -e
cd $PROJECT/$USER/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
N=${N:-8000}
[ -f solution/cache/feat_test.npz ] || python solution/features.py --device cuda
# one synthesis process per method, one GPU each (seed 1 -> different samples from the laptop run)
CUDA_VISIBLE_DEVICES=0 python solution/synth.py --method pixelseal  --n $N --device cuda --threads 16 --seed 1 --out solution/synth_big &
CUDA_VISIBLE_DEVICES=1 python solution/synth.py --method trustmark  --n $N --device cuda --threads 16 --seed 1 --out solution/synth_big &
CUDA_VISIBLE_DEVICES=2 python solution/synth.py --method stegastamp --n $N --device cuda --threads 16 --seed 1 --out solution/synth_big &
wait
CUDA_VISIBLE_DEVICES=0 python solution/train_seg.py --device cuda --epochs ${EPOCHS:-20} --bs 32 --lr 3e-3 \
    --synth solution/synth_big --out solution/ckpt/seg_big.pt
CUDA_VISIBLE_DEVICES=0 python solution/make_submission.py --ckpt solution/ckpt/seg_big.pt --device cuda --out solution/submission_big.pt
