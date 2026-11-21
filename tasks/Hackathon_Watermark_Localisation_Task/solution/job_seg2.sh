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
run 0 trustmark  trustmark  --arch feat &
run 1 pixelseal  pixelseal  --arch feat --amp 0 &
run 2 stegastamp stegastamp --arch pix &
run 3 pixelseal_b pixelseal --arch feat --seed 1 --bb-lr 0.03 --amp 0 &
wait
[ ! -f $O/failed.txt ]
