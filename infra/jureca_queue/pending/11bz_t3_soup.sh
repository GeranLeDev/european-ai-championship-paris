#!/bin/bash
# Task 3: model soup (uniform + greedy on proxy mixcls) of mixcls runs fine-tuned from the same orig
cd unlearning && source .venv/bin/activate && cd solution
python -u soup.py --device cuda --key mixcls --runs m1_g2 m2_g2 m2_g3 r2_g2 'ma_g*' 'mb_g*' 2>&1 | grep -v Warning
