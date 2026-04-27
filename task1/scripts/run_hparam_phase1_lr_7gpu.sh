#!/usr/bin/env bash
set -e

# Run from task1/:
#   bash scripts/run_hparam_phase1_lr_7gpu.sh

mkdir -p logs/hparam
mkdir -p outputs/hparam_plots

CONFIG="configs/resnet18_pretrained.yaml"

CUDA_VISIBLE_DEVICES=0 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p1_lr_3e-5_3e-4 training.backbone_lr=3e-5 training.head_lr=3e-4 \
  > "logs/hparam/hparam_p1_lr_3e-5_3e-4.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p1_lr_5e-5_5e-4 training.backbone_lr=5e-5 training.head_lr=5e-4 \
  > "logs/hparam/hparam_p1_lr_5e-5_5e-4.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p1_lr_1e-4_1e-3 training.backbone_lr=1e-4 training.head_lr=1e-3 \
  > "logs/hparam/hparam_p1_lr_1e-4_1e-3.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p1_lr_2e-4_2e-3 training.backbone_lr=2e-4 training.head_lr=2e-3 \
  > "logs/hparam/hparam_p1_lr_2e-4_2e-3.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p1_lr_3e-4_3e-3 training.backbone_lr=3e-4 training.head_lr=3e-3 \
  > "logs/hparam/hparam_p1_lr_3e-4_3e-3.log" 2>&1 &

CUDA_VISIBLE_DEVICES=5 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p1_lr_1e-4_5e-4 training.backbone_lr=1e-4 training.head_lr=5e-4 \
  > "logs/hparam/hparam_p1_lr_1e-4_5e-4.log" 2>&1 &

CUDA_VISIBLE_DEVICES=6 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p1_lr_1e-4_2e-3 training.backbone_lr=1e-4 training.head_lr=2e-3 \
  > "logs/hparam/hparam_p1_lr_1e-4_2e-3.log" 2>&1 &

wait
echo "[OK] Phase 1 done. Logs are under logs/hparam/."

