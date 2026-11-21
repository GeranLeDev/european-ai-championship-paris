#!/bin/bash
P=../.venv/bin/python
run(){ $P -u feat_unlearn.py --level mlp "$@" 2>&1 | grep -v Warn >> runs/mlp_grid2.log; }
run --method td --target medium --alpha 10 --lr 1e-3 --w-ce 1.0 --steps 800
run --method td --target medium --alpha 10 --lr 1e-3 --w-ce 0.3 --steps 1200
run --method td --target strong --alpha 10 --lr 1e-3 --w-ce 0.3 --steps 800
run --method td --target strong --alpha 30 --lr 1e-3 --w-ce 0.1 --steps 1200
run --method td --target cls --alpha 10 --lr 1e-3 --w-ce 0.2 --steps 800
run --method td --target cls --alpha 30 --lr 1e-3 --w-ce 0.05 --steps 1200
run --method td --target mix --select strong --alpha 10 --lr 1e-3 --w-ce 0.1 --steps 800
run --method td --target mix --select medium --alpha 10 --lr 1e-3 --w-ce 0.3 --steps 800
run --method td --target mixcls --select cls --alpha 10 --lr 1e-3 --w-ce 0.1 --steps 800
echo DONE >> runs/mlp_grid2.log
