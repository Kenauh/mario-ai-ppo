#!/usr/bin/env bash
# Cola: cfg1 desde cero hasta 5M; luego cfg2 de 2M a 5M (continuacion con --resume).
# Ctrl+C detiene TODA la cola (trap); un fallo de cfg1 tambien.
set -o pipefail
trap 'echo "Cola interrumpida"; exit 130' INT
cd "$(dirname "${BASH_SOURCE[0]}")/.." && source .venv/bin/activate
python -u scripts/train.py --config configs/base.json --total-steps 5000000 2>&1 | tee runs/cfg1_stdout.log || exit 1
python -u scripts/train.py --config configs/cfg2_reward_scale.json --resume --total-steps 5000000 2>&1 | tee -a runs/cfg2_stdout.log
