#!/usr/bin/env bash
# Reanuda cfg1 (desde su ultimo ckpt) hasta 5M y luego cfg2 hasta 5M. Ctrl+C detiene todo.
set -o pipefail
trap 'echo "Cola interrumpida"; exit 130' INT
cd ~/mario-ai && source .venv/bin/activate
python -u scripts/train.py --config configs/base.json --resume --total-steps 5000000 2>&1 | tee -a runs/cfg1_stdout.log || exit 1
python -u scripts/train.py --config configs/cfg2_reward_scale.json --resume --total-steps 5000000 2>&1 | tee -a runs/cfg2_stdout.log
