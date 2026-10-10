#!/usr/bin/env bash
#   bash run_sweep.sh
#   bash run_sweep.sh --epochs 10000

set -euo pipefail

SEEDS=(1 2 3 4 5)
WDS=( 0.55 0.56 )
P=71

mkdir -p logs

total=$(( ${#SEEDS[@]} * ${#WDS[@]} ))
i=0
for wd in "${WDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    i=$((i + 1))
    echo "=== [$i/$total] p=$P wd=$wd seed=$seed ==="
    python train.py --p "$P" --wd "$wd" --seed "$seed" "$@" --epochs 60000 \
      2>&1 | tee "logs/p${P}_wd${wd}_seed${seed}.log"
  done
done

echo "Gotowe: $total przebiegów."