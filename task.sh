#!/usr/bin/env bash
set -euo pipefail

SEEDS=(31 317 31731)
NUM_EP="${NUM_EP:-301}"
LR="${LR:-3e-3}"

run_train() {
  local title="$1"
  local seed="$2"
  shift 2

  printf '\n%s | seed=%s | num_ep=%s\n' "$title" "$seed" "$NUM_EP"
  uv run oneshot train \
    --dataset rmnist \
    --model lenet \
    --seed "$seed" \
    --lr "$LR" \
    --num_ep "$NUM_EP" \
    --gan_ratio 0 \
    "$@"
}

for seed in "${SEEDS[@]}"; do
  run_train "Baseline LeNet with Dropout" "$seed" --no-data-augmentation
done

for seed in "${SEEDS[@]}"; do
  run_train "LeNet with data_augmentation (1024)" "$seed" --data-augmentation
done