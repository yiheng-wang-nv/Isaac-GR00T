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
    --output_dir /localhome/local-vennw/code/Isaac-GR00T/outputs/sim_stage1 \
    --save_steps 10000 \
    --save_total_limit 5 \
    --max_steps 20000 \
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
    --background-noise-on-mask

# visualize the noise transform
python /localhome/local-vennw/code/Isaac-GR00T/scripts/dump_dataloader_transforms.py \
  --dataset_path \
    /localhome/local-vennw/code/orca-sim-pick-and-place-mimic/stage1_3_cosmos/lerobot/ \
    /localhome/local-vennw/code/orca-sim-pick-and-place-mimic/stage1_5_cosmos/lerobot/ \
    /localhome/local-vennw/code/orca-sim-pick-and-place-mimic/stage1_7_cosmos/lerobot/ \
    /localhome/local-vennw/code/orca-sim-pick-and-place-mimic/stage1_8_cosmos/lerobot/ \
  --embodiment_tag NEW_EMBODIMENT \
  --modality_config_path /localhome/local-vennw/code/Isaac-GR00T/orca_g1_locomanip_modality_config.py \
  --background_noise_on_mask \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --output_dir /localhome/local-vennw/code/Isaac-GR00T/debug_dataloader_transforms \
  --comparison_per_dataset \
  --comparison_episode_index 0