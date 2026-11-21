#!/bin/bash
#SBATCH --account=training2666
#SBATCH --gres=gpu:1
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=01:30:00
#SBATCH --output=output/%j.out
# Usage (from <team>/unlearning/solution, venv of the task activated, model.pt + forget_images.npz in ..):
#   mkdir -p output && sbatch jureca_main.sh
# Needs Imagenette (retain proxy) in data/imagenette2-320 (curl https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz)
set -e
python inspect_orig.py --device cuda
python gold_proxy.py --device cuda --k 8
python prep_retain.py
# chosen config (see challenge/task3_unlearning_report.md)
srun python unlearn.py --device cuda --amp --method td --target medium --blocks 12 --steps 600 --lr 2e-5 \
     --bs-f 64 --bs-r 64 --eval-every 50 --n-eval 3500 --n-eval-r 1000 --out runs/td_medium_full
python make_submission.py runs/td_medium_full/best.pt ../submission_td_medium.pt
