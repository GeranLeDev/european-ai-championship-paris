#!/bin/bash
# Third round (queue task, no sbatch), team convention WM_VAL=odd. Tests the "image x2" lever of the Modal pipeline on
# our extractor models: the extractor also sees the image at 1024 (its stride-4 features are then 256 x 256).
# Resumable, probabilities exported every 1000 steps and at the end, failed writes are not fatal.
cd /p/scratch/training2666/geran1/IncomingPHd/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export HF_HUB_OFFLINE=1 WM_VAL=odd
MIN=${MIN:-60}; O=output/seg3; mkdir -p $O output/probs
run() { # gpu name method extra-args
  CUDA_VISIBLE_DEVICES=$1 python solution/train_seg2.py --arch feat --method $3 --device cuda --data release --out $O/$2 --minutes $MIN --threads 10 \
    --probs-dir output/probs --pipeline jureca_$2 --export-every 1000 --p-real-cover 0.4 ${@:4} >> $O/$2.log 2>&1 || echo "FAILED $2" >> $O/failed4.txt
}
run 0 tmx trustmark --scales 512,1024 &
run 1 tmb trustmark --seed 1 --scales 256,512,1024 --bs 12 --n-real 3 &
run 2 psx pixelseal --amp 0 --scales 512,1024 --bs 8 --n-real 2 &
run 3 stx stegastamp --scales 400,800 &
wait
[ ! -f $O/failed4.txt ]
