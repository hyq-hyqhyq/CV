#!/usr/bin/env bash
set -e

# Run from task1/:
#   bash scripts/run_hparam_phase3_final_7gpu.sh
# You can override best values from env, e.g.:
#   BEST_BACKBONE_LR=2e-4 BEST_HEAD_LR=2e-3 BEST_EPOCHS=30 BEST_WEIGHT_DECAY=5e-4 bash scripts/run_hparam_phase3_final_7gpu.sh

BEST_BACKBONE_LR=${BEST_BACKBONE_LR:-1e-4}
BEST_HEAD_LR=${BEST_HEAD_LR:-1e-3}
BEST_EPOCHS=${BEST_EPOCHS:-20}
BEST_WEIGHT_DECAY=${BEST_WEIGHT_DECAY:-1e-4}

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

run_one 0 hparam_p3_final_lr_ep \
  training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" training.epochs="$BEST_EPOCHS"

run_one 1 hparam_p3_final_lr_wd \
  training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" training.weight_decay="$BEST_WEIGHT_DECAY"

run_one 2 hparam_p3_final_lr_dropout \
  training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" model.dropout=0.2

run_one 3 hparam_p3_final_lr_ls \
  training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" training.label_smoothing=0.1

run_one 4 hparam_p3_final_lr_ep_wd \
  training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" training.epochs="$BEST_EPOCHS" training.weight_decay="$BEST_WEIGHT_DECAY"

run_one 5 hparam_p3_final_lr_ep_wd_dropout \
  training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" training.epochs="$BEST_EPOCHS" training.weight_decay="$BEST_WEIGHT_DECAY" model.dropout=0.2

run_one 6 hparam_p3_final_lr_ep_wd_ls \
  training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" training.epochs="$BEST_EPOCHS" training.weight_decay="$BEST_WEIGHT_DECAY" training.label_smoothing=0.1

FAILED=0
if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "[OK] Phase 3 already complete."
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
  echo "[ERROR] Phase 3 finished with failures. See logs/hparam/*.log"
  exit 1
fi

echo "[OK] Phase 3 done. Logs are under logs/hparam/."
