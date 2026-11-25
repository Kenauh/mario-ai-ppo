#!/usr/bin/env bash
# Fase 7: evaluacion estandarizada de 4 checkpoints (2M y 5M de cfg1 y cfg2), protocolos P1 y P2.
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." && source .venv/bin/activate
export OMP_NUM_THREADS=1
mkdir -p results/eval
run() { python -u scripts/eval_standard.py --ckpt "$1" --label "$2" --mode "$3" --episodes "$4" --noop-max "$5" --out "results/eval/$2_$3_n$5.csv" 2>&1 | grep -v -i "warn" | grep -v -E "Gym has|Please upgrade|Users of|migration guide"; }
for spec in "cfg1_2M models/cfg1_base/ckpt_2000000_steps.zip" "cfg2_2M models/cfg2_reward_scale/ckpt_2000000_steps.zip" "cfg1_5M models/cfg1_base/ckpt_5000256_steps.zip" "cfg2_5M models/cfg2_reward_scale/ckpt_5000192_steps.zip"; do
  set -- $spec
  run "$2" "$1" stoch 100 0
  run "$2" "$1" det 1 0
  run "$2" "$1" stoch 100 8
  run "$2" "$1" det 9 8
done
