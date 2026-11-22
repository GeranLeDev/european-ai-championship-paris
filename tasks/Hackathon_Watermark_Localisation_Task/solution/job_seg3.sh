#!/bin/bash
# Second training round (queue task, no sbatch), team validation convention (WM_VAL=odd: no odd official image is used
# for training nor as cover; validation = odd half). Validated architecture only (official extractor fine-tuned + head).
# Probabilities exported in the exchange format every 1000 steps and at the end (same files rewritten; the node stops 03:32).
cd /p/scratch/training2666/geran1/IncomingPHd/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export HF_HUB_OFFLINE=1 WM_VAL=odd
MIN=${MIN:-50}; O=output/seg3; mkdir -p $O output/probs
run() { # gpu name method extra-args
  CUDA_VISIBLE_DEVICES=$1 python solution/train_seg2.py --arch feat --method $3 --device cuda --data release --out $O/$2 --minutes $MIN --threads 10 \
    --probs-dir output/probs --pipeline jureca_$2 --export-every 1000 --p-real-cover 0.4 ${@:4} > $O/$2.log 2>&1 || echo "FAILED $2" >> $O/failed.txt
}
run 0 psa pixelseal --amp 0 &
run 1 psb pixelseal --amp 0 --seed 1 --bb-lr 0.3 &
run 2 tma trustmark &
run 3 psc pixelseal --amp 0 --seed 2 --bb-lr 0.03 --p-real-cover 0.2 &
wait
[ ! -f $O/failed.txt ]
