#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=02:00:00
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
# 4 full-model configs in parallel (one per GPU). Usage: sbatch solution/jureca_grid.sh <tag> "<cfg0>" "<cfg1>" "<cfg2>" "<cfg3>"
cd /p/scratch/training2666/geran1/IncomingPHd/unlearning && source .venv/bin/activate
cd solution
TAG=$1; shift
i=0
for cfg in "$@"; do
  CUDA_VISIBLE_DEVICES=$i python -u unlearn.py --device cuda --amp $cfg --out ../output/runs/${TAG}_g$i > ../output/${TAG}_g$i.log 2>&1 &
  i=$((i+1))
done
wait
