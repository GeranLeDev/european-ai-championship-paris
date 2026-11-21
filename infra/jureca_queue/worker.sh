#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --output=/p/scratch/training2666/geran1/IncomingPHd/queue/logs/worker_%j.out
# Persistent worker: holds its node and runs queued tasks one after another (each task gets the whole 4-GPU node).
# Enqueue a task: write a bash script to queue/pending/<prio>_<name>.sh (lowest name first). It runs with cwd=IncomingPHd.
Q=/p/scratch/training2666/geran1/IncomingPHd/queue
cd /p/scratch/training2666/geran1/IncomingPHd
source ~/.bashrc >/dev/null 2>&1
echo "worker $SLURM_JOB_ID on $(hostname) start $(date)"
while true; do
  t=$(ls $Q/pending/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$t" ]; then sleep 20; continue; fi
  b=$(basename $t .sh); r=$Q/running/${b}.w$SLURM_JOB_ID.sh
  mv "$t" "$r" 2>/dev/null || continue        # atomic claim; another worker may have taken it
  echo "$(date +%T) START $b"
  ( cd /p/scratch/training2666/geran1/IncomingPHd && bash "$r" ) > $Q/logs/${b}.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then mv "$r" $Q/done/; else mv "$r" $Q/failed/; fi
  echo "$(date +%T) END $b rc=$rc"
done
