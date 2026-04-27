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

CUDA_VISIBLE_DEVICES=0 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p2_bestlr_ep10 training.epochs=10 training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" \
  > "logs/hparam/hparam_p2_bestlr_ep10.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p2_bestlr_ep20 training.epochs=20 training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" \
  > "logs/hparam/hparam_p2_bestlr_ep20.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p2_bestlr_ep30 training.epochs=30 training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" \
  > "logs/hparam/hparam_p2_bestlr_ep30.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p2_bestlr_wd_5e-5 training.weight_decay=5e-5 training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" \
  > "logs/hparam/hparam_p2_bestlr_wd_5e-5.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p2_bestlr_wd_5e-4 training.weight_decay=5e-4 training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" \
  > "logs/hparam/hparam_p2_bestlr_wd_5e-4.log" 2>&1 &

CUDA_VISIBLE_DEVICES=5 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p2_bestlr_dropout_0p2 model.dropout=0.2 training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" \
  > "logs/hparam/hparam_p2_bestlr_dropout_0p2.log" 2>&1 &

CUDA_VISIBLE_DEVICES=6 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p2_bestlr_ls_0p1 training.label_smoothing=0.1 training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" \
  > "logs/hparam/hparam_p2_bestlr_ls_0p1.log" 2>&1 &

wait
echo "[OK] Phase 2 done. Logs are under logs/hparam/."

