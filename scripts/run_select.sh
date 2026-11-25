#!/usr/bin/env bash
# Seleccion de checkpoint de cfg2: P2 estocastico, semillas 0 a 99 (conjunto de seleccion).
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." && source .venv/bin/activate
export OMP_NUM_THREADS=1
f() { grep -v -i "warn" | grep -v -E "Gym has|Please upgrade|Users of|migration guide"; }
for s in 3000 4000 4500; do
  ck=$(ls models/cfg2_reward_scale/ckpt_${s}896_steps.zip | head -1); echo "candidato: $ck"
  python -u scripts/eval_standard.py --ckpt "$ck" --label cfg2_${s}k --mode stoch --episodes 100 --noop-max 8 --out results/eval/sel_cfg2_${s}k_stoch_n8.csv 2>&1 | f
done
python -u scripts/eval_standard.py --ckpt models/cfg2_reward_scale/best.zip --label cfg2_best --mode stoch --episodes 100 --noop-max 8 --out results/eval/sel_cfg2_best_stoch_n8.csv 2>&1 | f
