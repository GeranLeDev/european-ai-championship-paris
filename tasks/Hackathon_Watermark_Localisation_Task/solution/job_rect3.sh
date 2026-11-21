#!/bin/bash
# Rectangle search, reduced + rebalanced for the single shared node (queue task, no sbatch). Snapshot of every cache
# written so far -> solution/cache/partial (never searched again); new caches: tag _r2 (train) / _t_r2 (test).
# TrustMark test (full grid), PixelSeal train + test (half-density grid), StegaStamp train: half of the remaining
# images only (58 s/image, used to decide whether a StegaStamp test run is worth the node time).
cd /p/scratch/training2666/geran1/IncomingPHd/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
P=solution/cache/partial; mkdir -p $P
cp -p solution/cache/search_*.npz $P/
GF=24,1.3,0.25,2.0; GR=${GR:-24,1.4,0.33,2.0}
s() { # method split shard grid bs
  local tag=_r2; [ $2 = test ] && tag=_t_r2
  python solution/run_search.py --data release --method $1 --split $2 --device cuda --shard $3 --grid $4 --bs $5 --tag $tag --skip-dir $P || echo "FAILED $1 $2 $3"
}
gpu() { export CUDA_VISIBLE_DEVICES=$1; shift; for j in "$@"; do echo "=== $(date +%T) $j"; s $j; done; echo "=== $(date +%T) end"; }
gpu 0 "trustmark test 0/1 $GF 256" "pixelseal test 0/6 $GR 256" > output/rect3_gpu0.log 2>&1 &
gpu 1 "pixelseal train 0/1 $GR 256" "pixelseal test 1/6 $GR 256" "pixelseal test 2/6 $GR 256" > output/rect3_gpu1.log 2>&1 &
gpu 2 "stegastamp train 0/4 $GR 128" "pixelseal test 3/6 $GR 256" "pixelseal test 4/6 $GR 256" > output/rect3_gpu2.log 2>&1 &
gpu 3 "stegastamp train 2/4 $GR 128" "pixelseal test 5/6 $GR 256" > output/rect3_gpu3.log 2>&1 &
wait
! grep -q "^FAILED" output/rect3_gpu*.log
