#!/bin/bash
set -x -e

# Fine-tuning script for Unitree G1 + Dex3 trocar assembly dataset
# Dataset: install_trocar_from_tray_realsense_lerobot
# Multiple training runs with different max_steps for comparison

export NUM_GPUS=2

# Path configuration
DATASET_PATH="/localhome/local-vennw/code/datasets/install_trocar_from_tray_realsense_lerobot"
MODALITY_CONFIG_PATH="unitree_g1_dex3_config.py"
BASE_OUTPUT_DIR="outputs/install_trocar_model"

# Run from Isaac-GR00T directory
cd /localhome/local-vennw/code/Isaac-GR00T

# Define different max_steps for comparison experiments
MAX_STEPS_LIST=(10000 20000 50000)

for MAX_STEPS in "${MAX_STEPS_LIST[@]}"; do
    echo "=========================================="
    echo "Training with max_steps=${MAX_STEPS}"
    echo "=========================================="
    
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_steps${MAX_STEPS}"
    
    # Calculate save_steps (save 5 checkpoints evenly)
    SAVE_STEPS=$((MAX_STEPS / 5))
    
    torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 \
        gr00t/experiment/launch_finetune.py \
        --base_model_path nvidia/GR00T-N1.6-3B \
        --dataset_path ${DATASET_PATH} \
        --modality_config_path ${MODALITY_CONFIG_PATH} \
        --embodiment_tag NEW_EMBODIMENT \
        --num_gpus ${NUM_GPUS} \
        --output_dir ${OUTPUT_DIR} \
        --save_steps ${SAVE_STEPS} \
        --save_total_limit 5 \
        --max_steps ${MAX_STEPS} \
        --warmup_ratio 0.05 \
        --weight_decay 1e-5 \
        --learning_rate 1e-4 \
        --global_batch_size 64 \
        --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
        --dataloader_num_workers 4
    
    echo "Finished training with max_steps=${MAX_STEPS}"
    echo ""
done

echo "All training runs completed!"

