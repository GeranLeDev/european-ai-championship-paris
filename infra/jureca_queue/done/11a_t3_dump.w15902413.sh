#!/bin/bash
# Task 3: dump eval logits of all checkpoints for offline hypothesis fitting
cd unlearning && source .venv/bin/activate && cd solution && python -u dump_logits.py ../output/runs/*/best.pt
