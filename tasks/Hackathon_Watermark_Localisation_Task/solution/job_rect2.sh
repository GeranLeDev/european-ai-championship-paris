#!/bin/bash
# Rectangle search, reduced version for the single shared node (queue task, no sbatch): resumes from the caches
# already written (snapshot in solution/cache/partial), 4 GPUs fully used, half-density grid for PixelSeal and
# StegaStamp (decoder-only ONNX graph). New caches: tag _r (train) and _t_r (test), one file per shard.
cd /p/scratch/training2666/geran1/IncomingPHd/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
P=solution/cache/partial; mkdir -p $P
cp -p solution/cache/search_train_pixelseal.npz solution/cache/search_train_trustmark.npz $P/ 2>/dev/null
GF=24,1.3,0.25,2.0; GR=${GR:-24,1.4,0.33,2.0}
s() { # method split shard grid bs
  local tag=_r; [ $2 = test ] && tag=_t_r
  python solution/run_search.py --data release --method $1 --split $2 --device cuda --shard $3 --grid $4 --bs $5 --tag $tag --skip-dir $P || echo "FAILED $1 $2 $3"
}
gpu() { export CUDA_VISIBLE_DEVICES=$1; shift; for j in "$@"; do echo "=== $(date +%T) $j"; s $j; done; echo "=== $(date +%T) end"; }
gpu 0 "trustmark test 0/1 $GF 256" "pixelseal test 0/4 $GR 256" "pixelseal train 0/2 $GR 256" "stegastamp train 1/4 $GR 128" > output/rect2_gpu0.log 2>&1 &
gpu 1 "pixelseal train 1/2 $GR 256" "pixelseal test 1/4 $GR 256" "pixelseal test 2/4 $GR 256" "stegastamp train 0/4 $GR 128" > output/rect2_gpu1.log 2>&1 &
gpu 2 "stegastamp train 2/4 $GR 128" "stegastamp test 0/4 $GR 128" "stegastamp test 1/4 $GR 128" "stegastamp train 3/4 $GR 128" > output/rect2_gpu2.log 2>&1 &
gpu 3 "pixelseal test 3/4 $GR 256" "stegastamp test 2/4 $GR 128" "stegastamp test 3/4 $GR 128" > output/rect2_gpu3.log 2>&1 &
wait
! grep -q "^FAILED" output/rect2_gpu*.log
