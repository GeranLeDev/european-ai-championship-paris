#!/bin/bash
# Catch-up after the 20:46 disk-quota incident: only the PixelSeal clean-mode test images that have no search result
# yet (snapshot of all caches -> partial, --margin 0), 4 shards on 4 GPUs. New tag _t_r3. ~7 min of node time.
cd /p/scratch/training2666/geran1/IncomingPHd/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
P=solution/cache/partial; mkdir -p $P
cp -p solution/cache/search_*.npz $P/
for g in 0 1 2 3; do
  ( CUDA_VISIBLE_DEVICES=$g python solution/run_search.py --data release --method pixelseal --split test --device cuda --shard $g/4 \
      --grid 24,1.4,0.33,2.0 --bs 256 --tag _t_r3 --margin 0 --skip-dir $P || echo "FAILED pixelseal test $g/4" ) > output/rect4_gpu$g.log 2>&1 &
done
wait
! grep -q "^FAILED" output/rect4_gpu*.log
