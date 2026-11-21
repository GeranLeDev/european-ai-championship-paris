#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=05:00:00
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
# Rectangle search (crop -> embed -> paste exploit), train then test, one GPU per method (stegastamp on 2 GPUs).
cd /p/scratch/training2666/geran1/IncomingPHd/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
G=${GRID:-24,1.3,0.25,2.0}; SPLITS=${SPLITS:-train test}; TAG=${TAG:-}
run() { # gpu method shard
  for sp in $SPLITS; do
    CUDA_VISIBLE_DEVICES=$1 python solution/run_search.py --data release --method $2 --split $sp --device cuda --shard $3 --grid $G --tag "$TAG"
  done > output/rect${TAG}_${SPLITS// /-}_$2_${3/\//of}.log 2>&1
}
run 0 trustmark 0/1 &
run 1 pixelseal 0/1 &
run 2 stegastamp 0/2 &
run 3 stegastamp 1/2 &
wait
