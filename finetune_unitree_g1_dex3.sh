#!/bin/bash
set -x -e

# Fine-tuning script for Unitree G1 + Dex3 trocar assembly dataset
# Dataset: install_trocar_from_tray_realsense_lerobot

export NUM_GPUS=1

# Path configuration
DATASET_PATH="/localhome/local-vennw/code/datasets/install_trocar_from_tray_realsense_lerobot"
MODALITY_CONFIG_PATH="${DATASET_PATH}/unitree_g1_dex3_config.py"
OUTPUT_DIR="/localhome/local-vennw/code/outputs/unitree_g1_dex3_finetune"

# Run from Isaac-GR00T directory
cd /localhome/local-vennw/code/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0 python \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path ${DATASET_PATH} \
    --modality_config_path ${MODALITY_CONFIG_PATH} \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus ${NUM_GPUS} \
    --output_dir ${OUTPUT_DIR} \
    --save_steps 1000 \
    --save_total_limit 5 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 32 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4

