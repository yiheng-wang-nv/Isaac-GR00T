#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t



CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /localcode/orca-template1-dev/task7_20260122_trimmed_with_mask \
  --embodiment-tag NEW_EMBODIMENT \
  --num-gpus 4 \
  --output-dir outputs/task7_20260122_trimmed_with_mask_4gpu \
  --save-total-limit 5 \
  --save-steps 5000 \
  --max-steps 20000 \
  --global-batch-size 768 \
  --learning-rate 1e-4 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-5 \
  --use_wandb \
  --gradient_accumulation_steps 1 \
  --dataloader-num-workers 4 --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --extra_augmentation_config '{"background_noise_transforms": [{"target_mask_values": [0], "p": 0.95}]}'




python scripts/dump_dataloader_transforms.py \
  --dataset_path /localhome/local-vennw/code/orca-template1-dev/task7_20260122_trimmed_with_mask \
  --embodiment_tag NEW_EMBODIMENT \
  --output_dir /localhome/local-vennw/code/Isaac-GR00T/debug_augment_vis_real \
  --save_grid \
  --grid_num_frames 9 \
  --no_resize \
  --video_episode_index 0 --extra_augmentation_config '{"background_noise_transforms": [{"target_mask_values": [0], "p": 0.95}]}'


# docker

docker run -it --rm --gpus '"device=0,1,2,3"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65535:65535 \
    -v $(pwd)/..:/workspace/gr00t \
    -v /localhome/local-vennw/code:/localcode \
    gr00t-dev /bin/bash

source /workspace/gr00t/.venv/bin/activate
cd /workspace/gr00t
uv pip install transformers==4.51.3 --force-reinstall

