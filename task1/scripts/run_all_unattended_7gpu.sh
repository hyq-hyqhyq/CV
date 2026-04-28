#!/usr/bin/env bash
set -euo pipefail

# Run from task1/:
#   nohup bash scripts/run_all_unattended_7gpu.sh > logs/run_all_task1.log 2>&1 &

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs/model_cmp logs/hparam logs/run_all outputs/hparam_plots

echo "[INFO] Task1 root: $ROOT_DIR"
echo "[INFO] Python: $(command -v python)"
python - <<'PY'
import torch
print(f"[INFO] torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_devices={torch.cuda.device_count()}")
try:
    import timm
    print(f"[INFO] timm={timm.__version__}")
except Exception as exc:
    print(f"[WARN] timm import failed: {exc}")
PY

declare -a PIDS=()
declare -a NAMES=()

run_bg () {
  local gpu="$1"
  local name="$2"
  shift 2
  echo "[RUN] gpu=${gpu} name=${name} cmd=$*"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$@" \
    > "logs/run_all/${name}.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
}

wait_all () {
  local failed=0
  for i in "${!PIDS[@]}"; do
    local pid="${PIDS[$i]}"
    local name="${NAMES[$i]}"
    if ! wait "$pid"; then
      echo "[FAIL] ${name} failed. See logs/run_all/${name}.log"
      failed=1
    else
      echo "[OK] ${name}"
    fi
  done
  PIDS=()
  NAMES=()
  if [[ "$failed" -ne 0 ]]; then
    echo "[ERROR] Some jobs failed. Stop here so the bad logs are easy to inspect."
    exit 1
  fi
}

collect_and_plot () {
  python scripts/collect_hparam_results.py
  python scripts/plot_hparam_bar.py || true
}

best_phase1_lr () {
  python - <<'PY'
import csv
from pathlib import Path

summary = Path("outputs/hparam_summary.csv")
best = None
with summary.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        name = row.get("experiment_name", "")
        if not name.startswith("hparam_p1_lr_"):
            continue
        try:
            score = float(row["best_val_acc"])
            backbone_lr = row["backbone_lr"]
            head_lr = row["head_lr"]
        except (KeyError, TypeError, ValueError):
            continue
        if best is None or score > best[0]:
            best = (score, backbone_lr, head_lr, name)

if best is None:
    print("1e-4 1e-3")
else:
    print(best[1], best[2])
    print(f"[INFO] best_phase1={best[3]} best_val_acc={best[0]:.6f}", file=__import__("sys").stderr)
PY
}

best_phase2_values () {
  python - <<'PY'
import csv
from pathlib import Path

summary = Path("outputs/hparam_summary.csv")
best_epoch = None
best_wd = None
with summary.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        name = row.get("experiment_name", "")
        try:
            score = float(row["best_val_acc"])
        except (KeyError, TypeError, ValueError):
            continue
        if name.startswith("hparam_p2_bestlr_ep"):
            try:
                epochs = int(row["epochs"])
            except (TypeError, ValueError):
                continue
            if best_epoch is None or score > best_epoch[0]:
                best_epoch = (score, epochs, name)
        if name.startswith("hparam_p2_bestlr_wd"):
            try:
                weight_decay = float(row["weight_decay"])
            except (TypeError, ValueError):
                continue
            if best_wd is None or score > best_wd[0]:
                best_wd = (score, weight_decay, name)

epochs = best_epoch[1] if best_epoch else 20
weight_decay = best_wd[1] if best_wd else 1e-4
print(epochs, weight_decay)
if best_epoch:
    print(f"[INFO] best_phase2_epoch={best_epoch[2]} best_val_acc={best_epoch[0]:.6f}", file=__import__("sys").stderr)
if best_wd:
    print(f"[INFO] best_phase2_wd={best_wd[2]} best_val_acc={best_wd[0]:.6f}", file=__import__("sys").stderr)
PY
}

echo "===================="
echo "[STEP 1] Model comparison"
echo "===================="
run_bg 0 resnet18_pretrained python train.py --config configs/resnet18_pretrained.yaml
run_bg 1 resnet18_scratch python train.py --config configs/resnet18_scratch.yaml
run_bg 2 se_resnet18_pretrained python train.py --config configs/se_resnet18_pretrained.yaml
run_bg 3 cbam_resnet18_pretrained python train.py --config configs/cbam_resnet18_pretrained.yaml
run_bg 4 vit_tiny_pretrained python train.py --config configs/vit_tiny_pretrained.yaml
run_bg 5 swin_t_pretrained python train.py --config configs/swin_t_pretrained.yaml
wait_all
collect_and_plot

echo "===================="
echo "[STEP 2] Hyperparameter phase 1: learning rate search"
echo "===================="
bash scripts/run_hparam_phase1_lr_7gpu.sh
collect_and_plot

read BEST_BACKBONE_LR BEST_HEAD_LR < <(best_phase1_lr)
echo "[INFO] Using BEST_BACKBONE_LR=${BEST_BACKBONE_LR} BEST_HEAD_LR=${BEST_HEAD_LR}"

echo "===================="
echo "[STEP 3] Hyperparameter phase 2: epoch and regularization"
echo "===================="
BEST_BACKBONE_LR="$BEST_BACKBONE_LR" BEST_HEAD_LR="$BEST_HEAD_LR" \
  bash scripts/run_hparam_phase2_reg_epoch_7gpu.sh
collect_and_plot

read BEST_EPOCHS BEST_WEIGHT_DECAY < <(best_phase2_values)
echo "[INFO] Using BEST_EPOCHS=${BEST_EPOCHS} BEST_WEIGHT_DECAY=${BEST_WEIGHT_DECAY}"

echo "===================="
echo "[STEP 4] Hyperparameter phase 3: final combinations"
echo "===================="
BEST_BACKBONE_LR="$BEST_BACKBONE_LR" \
BEST_HEAD_LR="$BEST_HEAD_LR" \
BEST_EPOCHS="$BEST_EPOCHS" \
BEST_WEIGHT_DECAY="$BEST_WEIGHT_DECAY" \
  bash scripts/run_hparam_phase3_final_7gpu.sh
collect_and_plot

echo "===================="
echo "[DONE] All Task1 experiments finished"
echo "===================="
find outputs -type f \( \
  -name 'metrics.csv' -o \
  -name 'summary.json' -o \
  -name 'training_curves.png' -o \
  -name 'loss_curve.png' -o \
  -name 'accuracy_curve.png' -o \
  -name 'hparam_summary.csv' -o \
  -name '*.png' \
\) | sort
