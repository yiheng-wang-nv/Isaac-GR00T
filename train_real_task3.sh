#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t

# single gpu train
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /localhome/local-vennw/code/orca-template1-dev/task3_01210122_merged_with_mask \
  --embodiment-tag NEW_EMBODIMENT \
  --num-gpus 1 \
  --output-dir /localhome/local-vennw/code/Isaac-GR00T/outputs/task3_01210122_merged_with_mask \
  --save-total-limit 5 \
  --save-steps 5000 \
  --max-steps 150000 \
  --tune-visual \
  --global-batch-size 12 \
  --dataloader-num-workers 4 --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --extra_augmentation_config '{"background_noise_transforms": [{"target_mask_values": [0], "p": 0.95}]}'


# debug visualizations
python scripts/dump_dataloader_transforms.py \
  --dataset_path /localhome/local-vennw/code/orca-template1-dev/task3_01210122_merged_with_mask \
  --embodiment_tag NEW_EMBODIMENT \
  --output_dir /localhome/local-vennw/code/Isaac-GR00T/debug_augment_vis_real \
  --save_grid \
  --grid_num_frames 9 \
  --no_resize \
  --video_episode_index 0 --extra_augmentation_config '{"background_noise_transforms": [{"target_mask_values": [0], "p": 0.95}]}'
