#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t

CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /localhome/local-vennw/code/galbot_lerobot_dataset/task7_20260106_merged_lerobot \
  --embodiment-tag NEW_EMBODIMENT \
  --num-gpus 2 \
  --output-dir /localhome/local-vennw/code/Isaac-GR00T/outputs/task7_20260106_merged_lerobot \
  --save-total-limit 5 \
  --save-steps 5000 \
  --max-steps 50000 \
  --global-batch-size 32 \
  --dataloader-num-workers 4


CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /localhome/local-vennw/code/galbot_lerobot_dataset/task7_20260106_merged_lerobot \
  --embodiment-tag NEW_EMBODIMENT \
  --num-gpus 2 \
  --output-dir /localhome/local-vennw/code/Isaac-GR00T/outputs/task7_20260106_merged_lerobot_with_noise \
  --save-total-limit 5 \
  --save-steps 5000 \
  --max-steps 50000 \
  --background-noise-on-mask \
  --global-batch-size 32 \
  --dataloader-num-workers 4

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /localhome/local-vennw/code/galbot_lerobot_dataset/task7_20260106_merged_lerobot \
  --embodiment-tag NEW_EMBODIMENT \
  --num-gpus 2 \
  --output-dir /localhome/local-vennw/code/Isaac-GR00T/outputs/task7_20260106_merged_lerobot_with_noise_tune_visual_add_jitter \
  --save-total-limit 5 \
  --save-steps 5000 \
  --max-steps 50000 \
  --background-noise-on-mask \
  --tune-visual \
  --global-batch-size 32 \
  --dataloader-num-workers 4 --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /localhome/local-vennw/code/galbot_lerobot_dataset/task7_20260106_merged_lerobot \
  --embodiment-tag NEW_EMBODIMENT \
  --num-gpus 2 \
  --output-dir /localhome/local-vennw/code/Isaac-GR00T/outputs/task7_20260106_merged_lerobot_tune_visual_add_jitter \
  --save-total-limit 5 \
  --save-steps 10000 \
  --max-steps 50000 \
  --tune-visual \
  --global-batch-size 32 \
  --dataloader-num-workers 4 --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08