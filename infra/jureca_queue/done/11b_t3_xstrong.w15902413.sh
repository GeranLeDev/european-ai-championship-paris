#!/bin/bash
# Task 3: stronger gold proxy (public LB: strong 0.238 >> medium 0.080) -> probe further in that direction
cd unlearning && source .venv/bin/activate
cd solution && python -u gold_proxy.py --device cuda --k 8 --names xstrong && cd ..
bash solution/jureca_grid.sh rx "--method td --blocks 12 --bs-f 64 --bs-r 64 --eval-every 100 --n-eval 3500 --n-eval-r 1000 --target xstrong --alpha 10 --lr 5e-5 --steps 1500 --w-stat 0.1" "--method td --blocks 12 --bs-f 64 --bs-r 64 --eval-every 100 --n-eval 3500 --n-eval-r 1000 --target xstrong --alpha 10 --lr 5e-5 --steps 1500 --w-stat 0.3" "--method td --blocks 12 --bs-f 64 --bs-r 64 --eval-every 100 --n-eval 3500 --n-eval-r 1000 --target xstrong --alpha 20 --lr 5e-5 --steps 1500 --w-stat 0.1" "--method td --blocks 12 --bs-f 64 --bs-r 64 --eval-every 100 --n-eval 3500 --n-eval-r 1000 --target xstrong --alpha 10 --lr 1e-4 --steps 1000 --w-stat 0.1"
cd solution && for r in ../output/runs/rx_g*/; do [ -f $r/best.pt ] && python make_submission.py $r/best.pt ../output/sub_$(basename $r).pt; done; cd .. && python solution/collect.py
