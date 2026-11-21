#!/bin/bash
# Task 3 sweep x2 (public LB: mixcls 0.2616 > strong 0.2384): class-removal weight / offset / xstrong
cd unlearning && source .venv/bin/activate
for i in $(seq 1 90); do [ -f solution/cache/gold_xstrong.pt ] && break; sleep 20; done
bash solution/jureca_grid.sh x2 "--method td --blocks 12 --bs-f 64 --bs-r 64 --eval-every 100 --n-eval 3500 --n-eval-r 1000 --alpha 10 --lr 5e-5 --steps 1500 --w-stat 0.1 --target cls:xstrong:10" "--method td --blocks 12 --bs-f 64 --bs-r 64 --eval-every 100 --n-eval 3500 --n-eval-r 1000 --alpha 10 --lr 5e-5 --steps 1500 --w-stat 0.1 --target mix:cls:xstrong:10+strong" "--method td --blocks 12 --bs-f 64 --bs-r 64 --eval-every 100 --n-eval 3500 --n-eval-r 1000 --alpha 10 --lr 5e-5 --steps 1500 --w-stat 0.1 --target mix:cls:strong:10+xstrong" "--method td --blocks 12 --bs-f 64 --bs-r 64 --eval-every 100 --n-eval 3500 --n-eval-r 1000 --alpha 10 --lr 5e-5 --steps 1500 --w-stat 0.1 --target mix:cls:medium:10+xstrong"
cd solution && for r in ../output/runs/x2_g*/; do [ -f $r/best.pt ] && python make_submission.py $r/best.pt ../output/sub_$(basename $r).pt; done; cd .. && python solution/collect.py
