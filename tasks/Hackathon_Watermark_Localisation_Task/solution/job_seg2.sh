#!/bin/bash
# Trained segmenter (solution/train_seg2.py), queue task for the single shared node (no sbatch): one config per GPU.
# No dataset on disk (online generation); writes only checkpoints (~250 MB per config, atomic) and predictions.
# Resumable: re-queueing the same task continues from output/seg2/<name>/last.pt within the same time budget.
# MIN = training minutes per config (default 75) ; whole task = MIN + ~6 min (setup, final evaluation, test prediction).
cd /p/scratch/training2666/geran1/IncomingPHd/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
MIN=${MIN:-75}; O=output/seg2; mkdir -p $O
run() { # gpu name method extra-args
  CUDA_VISIBLE_DEVICES=$1 python solution/train_seg2.py --method $3 --device cuda --data release --out $O/$2 --minutes $MIN --threads 8 ${@:4} \
    > $O/$2.log 2>&1 || echo "FAILED $2" >> $O/failed.txt
}
rm -f $O/failed.txt
run 0 trustmark  trustmark  --use-what 0 &
run 1 pixelseal  pixelseal  --use-what 0 &
run 2 stegastamp stegastamp --use-what 0 &
run 3 trustmark_what trustmark --use-what 1 --seed 1 &
wait
[ ! -f $O/failed.txt ]
