#!/usr/bin/env bash
set -e

# Run from task1/:
#   bash scripts/run_hparam_phase2_reg_epoch_7gpu.sh
# Or override best LRs from env:
#   BEST_BACKBONE_LR=2e-4 BEST_HEAD_LR=2e-3 bash scripts/run_hparam_phase2_reg_epoch_7gpu.sh

BEST_BACKBONE_LR=${BEST_BACKBONE_LR:-1e-4}
BEST_HEAD_LR=${BEST_HEAD_LR:-1e-3}

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

run_one 0 hparam_p2_bestlr_ep10 \
  training.epochs=10 training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR"

run_one 1 hparam_p2_bestlr_ep20 \
  training.epochs=20 training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR"

run_one 2 hparam_p2_bestlr_ep30 \
  training.epochs=30 training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR"

run_one 3 hparam_p2_bestlr_wd_5e-5 \
  training.weight_decay=5e-5 training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR"

run_one 4 hparam_p2_bestlr_wd_5e-4 \
  training.weight_decay=5e-4 training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR"

run_one 5 hparam_p2_bestlr_dropout_0p2 \
  model.dropout=0.2 training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR"

run_one 6 hparam_p2_bestlr_ls_0p1 \
  training.label_smoothing=0.1 training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR"

FAILED=0
if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "[OK] Phase 2 already complete."
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
  echo "[ERROR] Phase 2 finished with failures. See logs/hparam/*.log"
  exit 1
fi

echo "[OK] Phase 2 done. Logs are under logs/hparam/."
