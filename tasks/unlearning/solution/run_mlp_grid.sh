#!/bin/bash
# grid of feature-level (last MLP + head) unlearning runs; one line per run in runs/mlp_grid.log
P=../.venv/bin/python
run(){ $P -u feat_unlearn.py --level mlp --steps 600 "$@" --json runs/mlp/$(echo "$@" | tr ' ' '_' | tr -d '-').json 2>&1 | grep -v Warn >> runs/mlp_grid.log; }
for t in medium strong cls; do
  for al in 10 30; do for lr in 3e-4 1e-3; do run --method td --target $t --alpha $al --lr $lr; done; done
done
run --method td --target medium --alpha 10 --lr 1e-3 --w-ce 0.3
run --method td --target strong --alpha 10 --lr 1e-3 --w-ce 0.1
run --method td --target cls --alpha 10 --lr 1e-3 --w-ce 0.05
run --method ga --target medium --alpha 10 --lr 3e-4 --cap 1.0
run --method ga --target strong --alpha 30 --lr 1e-4 --cap 3.0
run --method scrub --target medium --alpha 10 --lr 1e-4
run --method td --target soft:8 --select strong --alpha 10 --lr 1e-3
echo DONE >> runs/mlp_grid.log
