#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=02:00:00
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
# GPU0-2: sliding-window decoding maps (train+test) per method; GPU3: big synthetic set for stegastamp+trustmark
cd /p/scratch/training2666/geran1/IncomingPHd/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python solution/sliding.py --device cuda --methods pixelseal --splits train,test > output/slide_pixelseal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python solution/sliding.py --device cuda --methods trustmark --splits train,test > output/slide_trustmark.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python solution/sliding.py --device cuda --methods stegastamp --splits train,test > output/slide_stegastamp.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python solution/synth.py --method pixelseal --n 6000 --device cuda --threads 8 --seed 2 --out solution/synth_big > output/synth_big_pixelseal.log 2>&1 &
wait
