#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t

DATASET_PATH=/localhome/local-vennw/code/orca-sim-pick-and-place-mimic
CUDA_VISIBLE_DEVICES=2 python gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path  \
    $DATASET_PATH/stage1_3_cosmos/lerobot/ \
    $DATASET_PATH/stage1_5_cosmos/lerobot/ \
    $DATASET_PATH/stage1_7_cosmos/lerobot/ \
    $DATASET_PATH/stage1_8_cosmos/lerobot/ \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path /localhome/local-vennw/code/Isaac-GR00T/orca_g1_locomanip_modality_config.py  \
    --num_gpus 1 \
    --output_dir /localhome/local-vennw/code/Isaac-GR00T/outputs/sim_stage1_v2_50k \
    --save_steps 10000 \
    --save_total_limit 3 \
    --max_steps 50000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 32 \
    --tune_visual \
    --tune_projector \
    --tune_diffusion_model \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 8 \
    --extra_augmentation_config '{"background_noise_on_mask": 0.9, "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "grayscale_prob": 0.5, "alpha_range": [0, 1]}, {"target_mask_values": [5], "p": 1.0, "grayscale_prob": 0.5, "alpha_range": [0, 1]}]}'


DATASET_PATH=/localhome/local-vennw/code/orca-sim-pick-and-place-mimic
CUDA_VISIBLE_DEVICES=2 python gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path  \
    $DATASET_PATH/stage1_3_cosmos/lerobot/ \
    $DATASET_PATH/stage1_5_cosmos/lerobot/ \
    $DATASET_PATH/stage1_7_cosmos/lerobot/ \
    $DATASET_PATH/stage1_8_cosmos/lerobot/ \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path /localhome/local-vennw/code/Isaac-GR00T/orca_g1_locomanip_modality_config.py  \
    --num_gpus 1 \
    --output_dir /localhome/local-vennw/code/Isaac-GR00T/outputs/sim_stage1_v2_80k \
    --save_steps 10000 \
    --save_total_limit 3 \
    --max_steps 80000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 32 \
    --tune_visual \
    --tune_projector \
    --tune_diffusion_model \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 8 \
    --extra_augmentation_config '{"background_noise_on_mask": 0.9, "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "grayscale_prob": 0.5, "alpha_range": [0, 1]}, {"target_mask_values": [5], "p": 1.0, "grayscale_prob": 0.5, "alpha_range": [0, 1]}]}'



# debug - test full augmentation pipeline
python scripts/dump_dataloader_transforms.py \
  --dataset_path /localhome/local-vennw/code/orca-sim-pick-and-place-mimic/stage1_3_cosmos/lerobot/ \
  --embodiment_tag NEW_EMBODIMENT \
  --modality_config_path /localhome/local-vennw/code/Isaac-GR00T/orca_g1_locomanip_modality_config.py \
  --extra_augmentation_config '{"background_noise_on_mask": 0.9, "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "grayscale_prob": 0.5, "alpha_range": [0, 1]}, {"target_mask_values": [5], "p": 1.0, "grayscale_prob": 0.5, "alpha_range": [0, 1]}]}' \
  --output_dir /localhome/local-vennw/code/Isaac-GR00T/debug_full_augment \
  --save_video \
  --video_side_by_side \
  --video_episode_index 10 \
  --video_fps 10