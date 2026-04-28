#!/usr/bin/env bash
set -e

# Run from task1/:
#   bash scripts/run_hparam_phase1_lr_7gpu.sh

mkdir -p logs/hparam
mkdir -p outputs/hparam_plots

CONFIG="configs/resnet18_pretrained.yaml"

declare -a PIDS=()
declare -a NAMES=()

is_done () {
  local name="$1"
  find outputs -maxdepth 1 -type d -name "*_${name}" -exec test -f "{}/summary.json" \; -print -quit | grep -q .
}

run_one () {
  local gpu="$1"
  local name="$2"
  shift 2
  if is_done "$name"; then
    echo "[SKIP] ${name} already has summary.json"
    return
  fi
  echo "[RUN] gpu=${gpu} name=${name}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python train.py --config "$CONFIG" --set \
    "experiment.name=${name}" "$@" \
    > "logs/hparam/${name}.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${name}")
}

run_one 0 hparam_p1_lr_3e-5_3e-4 \
  training.backbone_lr=3e-5 training.head_lr=3e-4

run_one 1 hparam_p1_lr_5e-5_5e-4 \
  training.backbone_lr=5e-5 training.head_lr=5e-4

run_one 2 hparam_p1_lr_1e-4_1e-3 \
  training.backbone_lr=1e-4 training.head_lr=1e-3

run_one 3 hparam_p1_lr_2e-4_2e-3 \
  training.backbone_lr=2e-4 training.head_lr=2e-3

run_one 4 hparam_p1_lr_3e-4_3e-3 \
  training.backbone_lr=3e-4 training.head_lr=3e-3

run_one 5 hparam_p1_lr_1e-4_5e-4 \
  training.backbone_lr=1e-4 training.head_lr=5e-4

run_one 6 hparam_p1_lr_1e-4_2e-3 \
  training.backbone_lr=1e-4 training.head_lr=2e-3

FAILED=0
if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "[OK] Phase 1 already complete."
else
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    if ! wait "$pid"; then
      echo "[FAIL] ${name} (pid=${pid}). Check logs/hparam/${name}.log"
      FAILED=1
    else
      echo "[OK] ${name}"
    fi
  done
fi

if [[ "$FAILED" -ne 0 ]]; then
  echo "[ERROR] Phase 1 finished with failures. See logs/hparam/*.log"
  exit 1
fi

echo "[OK] Phase 1 done. Logs are under logs/hparam/."
