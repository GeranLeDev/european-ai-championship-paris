#!/bin/bash
# Task 3 sweep m3 (public LB: mixcls r2_g2 = 0.2616 best)
cd unlearning && source .venv/bin/activate
bash solution/jureca_grid.sh m3 "--method td --blocks 12 --bs-f 64 --bs-r 64 --eval-every 100 --n-eval 3500 --n-eval-r 1000 --target clsstrong --alpha 10 --lr 5e-5 --steps 1500 --w-stat 0.1" "--method td --blocks 12 --bs-f 64 --bs-r 64 --eval-every 100 --n-eval 3500 --n-eval-r 1000 --target clsstrong --alpha 20 --lr 5e-5 --steps 1500 --w-stat 0.05" "--method td --blocks 12 --bs-f 64 --bs-r 64 --eval-every 100 --n-eval 3500 --n-eval-r 1000 --target cls --alpha 10 --lr 5e-5 --steps 1500 --w-stat 0.1" "--method td --blocks 12 --bs-f 64 --bs-r 64 --eval-every 100 --n-eval 3500 --n-eval-r 1000 --target cls --alpha 20 --lr 5e-5 --steps 1500 --w-stat 0.05"
cd solution && for r in ../output/runs/m3_g*/; do [ -f $r/best.pt ] && python make_submission.py $r/best.pt ../output/sub_$(basename $r).pt; done; cd .. && python solution/collect.py
