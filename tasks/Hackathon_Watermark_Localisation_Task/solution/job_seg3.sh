#!/bin/bash
# Second training round (queue task, no sbatch), team validation convention (WM_VAL=odd: odd half of the official train
# images never used for training nor as covers). One config per GPU, resumable, raw probabilities exported to output/probs.
#   GPU0 trustmark : official extractor features + pretrained U-Net image branch
#   GPU1 pixelseal : same, float32 (its ConvNeXt-V2 extractor gives NaN in float16)
#   GPU2 stegastamp: teammate U-Net recipe, continues output/seg2/stegastamp_unet
#   GPU3 pixelseal : second seed, larger extractor learning rate
# MIN = training minutes per config (default 70); whole task = MIN + ~6 min.
cd /p/scratch/training2666/geran1/IncomingPHd/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export HF_HUB_OFFLINE=1 WM_VAL=odd
MIN=${MIN:-70}; O=output/seg3; mkdir -p $O output/probs
run() { # gpu name method minutes extra-args
  CUDA_VISIBLE_DEVICES=$1 python solution/train_seg2.py --method $3 --device cuda --data release --out $2 --minutes $4 --threads 10 \
    --probs-dir output/probs ${@:5} >> $2.log 2>&1 || echo "FAILED $2" >> $O/failed.txt
}
rm -f $O/failed.txt
run 0 $O/trustmark_fu trustmark $MIN --arch feat --pix 2 --p-real-cover 0.4 --pipeline jureca_fu &
run 1 $O/pixelseal_fu pixelseal $MIN --arch feat --pix 2 --amp 0 --p-real-cover 0.4 --pipeline jureca_fu &
run 2 output/seg2/stegastamp_unet stegastamp $((MIN + 38)) --arch unet --amp 0 --lr 3e-4 --bs 24 --n-real 6 --p-real-cover 0.5 --threads 12 --pipeline jureca_unet &
run 3 $O/pixelseal_fu_b pixelseal $MIN --arch feat --pix 2 --amp 0 --p-real-cover 0.4 --seed 1 --bb-lr 0.3 --pipeline jureca_fu_b &
wait
[ ! -f $O/failed.txt ]
