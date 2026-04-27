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

CUDA_VISIBLE_DEVICES=0 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p3_final_lr_ep training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" training.epochs="$BEST_EPOCHS" \
  > "logs/hparam/hparam_p3_final_lr_ep.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p3_final_lr_wd training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" training.weight_decay="$BEST_WEIGHT_DECAY" \
  > "logs/hparam/hparam_p3_final_lr_wd.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p3_final_lr_dropout training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" model.dropout=0.2 \
  > "logs/hparam/hparam_p3_final_lr_dropout.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p3_final_lr_ls training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" training.label_smoothing=0.1 \
  > "logs/hparam/hparam_p3_final_lr_ls.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p3_final_lr_ep_wd training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" training.epochs="$BEST_EPOCHS" training.weight_decay="$BEST_WEIGHT_DECAY" \
  > "logs/hparam/hparam_p3_final_lr_ep_wd.log" 2>&1 &

CUDA_VISIBLE_DEVICES=5 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p3_final_lr_ep_wd_dropout training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" training.epochs="$BEST_EPOCHS" training.weight_decay="$BEST_WEIGHT_DECAY" model.dropout=0.2 \
  > "logs/hparam/hparam_p3_final_lr_ep_wd_dropout.log" 2>&1 &

CUDA_VISIBLE_DEVICES=6 python train.py --config "$CONFIG" --set \
  experiment.name=hparam_p3_final_lr_ep_wd_ls training.backbone_lr="$BEST_BACKBONE_LR" training.head_lr="$BEST_HEAD_LR" training.epochs="$BEST_EPOCHS" training.weight_decay="$BEST_WEIGHT_DECAY" training.label_smoothing=0.1 \
  > "logs/hparam/hparam_p3_final_lr_ep_wd_ls.log" 2>&1 &

wait
echo "[OK] Phase 3 done. Logs are under logs/hparam/."

