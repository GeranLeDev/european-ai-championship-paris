#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=01:00:00
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
cd /p/scratch/training2666/geran1/IncomingPHd/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
C=64:32:1,64:32:3,32:16:3,128:32:2
CUDA_VISIBLE_DEVICES=0 python solution/blocktest.py --device cuda --methods trustmark --n 20 --configs $C > output/block_trustmark.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python solution/blocktest.py --device cuda --methods pixelseal --n 20 --configs $C > output/block_pixelseal.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python solution/blocktest.py --device cuda --methods stegastamp --n 20 --configs $C > output/block_stegastamp.log 2>&1 &
wait
