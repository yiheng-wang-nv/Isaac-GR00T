# Navigate to the GR00T repository
# If you are using the rheo workflow container with `-g1.5` flag, the path is `third_party/Isaac-GR00T`

source "/localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/third_party/cosmos-transfer2.5/.venv/bin/activate"

# full data base model
CUDA_VISIBLE_DEVICES=4,5,6,7 IS_TORCHRUN=1 \
torchrun --standalone --nproc_per_node=4 --nnodes=1 scripts/gr00t_finetune.py \
  --dataset-path /localhome/local-vennw/code/trocar_success_lt_7s_combined \
  --num-gpus 4 \
  --batch-size 64 \
  --output-dir sft_4gpu_256bs_50ksteps_success_lt7 \
  --data-config gr00t_config:UnitreeG1SimNoStageDataConfig \
  --video_backend decord \
  --report_to tensorboard \
  --max_steps 50000 \
  --save-steps 25000 --gradient_accumulation_steps 1

# full data + Cosmos background augmentation
# Templates pulled from HealthSurgiBench_1e4/media via DEFAULT_TEMPLATE_FOLDER in gr00t_mask_config.py
CUDA_VISIBLE_DEVICES=4,5,6,7 IS_TORCHRUN=1 \
torchrun --standalone --nproc_per_node=4 --nnodes=1 scripts/gr00t_finetune.py \
  --dataset-path /localhome/local-vennw/code/trocar_success_lt_7s_combined \
  --num-gpus 4 \
  --batch-size 64 \
  --output-dir sft_4gpu_256bs_50ksteps_success_lt7_bgaug \
  --data-config gr00t_mask_config:UnitreeG1SimMaskDataConfig \
  --video_backend decord \
  --report_to tensorboard \
  --max_steps 50000 \
  --save-steps 50000 \
  --gradient_accumulation_steps 1

# split-stage data + task-complete action supervision
# action dim = 28 robot dims + 1 action.task_complete soft label over the final 8 frames
CUDA_VISIBLE_DEVICES=0,1,2,3 IS_TORCHRUN=1 \
torchrun --standalone --nproc_per_node=4 --nnodes=1 scripts/gr00t_finetune.py \
  --dataset-path /localhome/local-vennw/code/trocar_success_lt_7s_split_by_stage_task_complete \
  --num-gpus 4 \
  --batch-size 64 \
  --output-dir sft_4gpu_256bs_50ksteps_split_stage_task_complete_soft8_bgaug \
  --data-config gr00t_mask_config:UnitreeG1SimMaskTaskCompleteDataConfig \
  --video_backend decord \
  --report_to tensorboard \
  --max_steps 100000 \
  --save-steps 50000 \
  --gradient_accumulation_steps 1

# split-stage data + task-complete action supervision, no background augmentation
# This is the direct ablation for the bgaug command above.
CUDA_VISIBLE_DEVICES=0,1,2,3 IS_TORCHRUN=1 \
torchrun --standalone --nproc_per_node=4 --nnodes=1 scripts/gr00t_finetune.py \
  --dataset-path /localhome/local-vennw/code/trocar_success_lt_7s_split_by_stage_task_complete \
  --num-gpus 4 \
  --batch-size 64 \
  --output-dir sft_4gpu_256bs_100ksteps_split_stage_task_complete_soft8_no_bgaug \
  --data-config gr00t_config:UnitreeG1SimTaskCompleteDataConfig \
  --video_backend decord \
  --report_to tensorboard \
  --max_steps 100000 \
  --save-steps 50000 \
  --gradient_accumulation_steps 1

# eval: SFT model, no mask aug
conda activate isaaclab_develop_6.0
CUDA_VISIBLE_DEVICES=1 python /localhome/local-vennw/code/IsaacLab/scripts/tools/record_trocar_episodes.py \
  --model_path /localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/sft_4gpu_256bs_50ksteps_success_lt7/checkpoint-50000 \
  --output_dir /localhome/local-vennw/code/sft_base_model_460data_noaug \
  --num_episodes 100 \
  --max_steps 512 \
  --use_gr00t_policy \
  --open_loop_steps 8 \
  --fixed_initial_state_dataset /localhome/local-vennw/code/trocar_success_lt_7s_combined \
  --fixed_initial_state_episode 0 \
  --fixed_initial_state_frame 0 \
  --fixed_initial_state_steps 30 \
  --fixed_initial_state_tolerance 0.035 --seed 42

# check multi-stage
DISPLAY=:1 XAUTHORITY=$HOME/.Xauthority \
  ./isaaclab.sh -p scripts/tools/interactive_trocar_multitask.py \
    --model_path /localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/sft_4gpu_256bs_100ksteps_split_stage_task_complete_soft8_no_bgaug/checkpoint-100000 \
    --device cuda:0 \
    --viz kit \
    --open_loop_steps 8 \
    --fixed_initial_state_dataset /localhome/local-vennw/code/trocar_success_lt_7s_combined \
    --fixed_initial_state_episode 0 \
    --fixed_initial_state_frame 0 \
    --fixed_initial_state_steps 30 \
    --fixed_initial_state_tolerance 0.035 --seed 42 --stage_pred_min_confidence 0.9 --retry_return_steps 10 --step_hz 10 --tray_yaw_increment_deg 5

# check task complete
DISPLAY=:1 XAUTHORITY=$HOME/.Xauthority \
  ./isaaclab.sh -p scripts/tools/interactive_trocar_task_complete.py \
    --model_path /localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/sft_4gpu_256bs_100ksteps_split_stage_task_complete_soft8_no_bgaug/checkpoint-100000 \
    --device cuda:0 \
    --open_loop_steps 8 \
    --fixed_initial_state_dataset /localhome/local-vennw/code/trocar_success_lt_7s_split_by_stage_task_complete \
    --fixed_initial_state_episode 0 \
    --fixed_initial_state_frame 0 \
    --fixed_initial_state_steps 10 \
    --fixed_initial_state_tolerance 0.035 --seed 42 --task_complete_threshold 0.9


# check demo
cd /localhome/local-vennw/code/IsaacLab
conda activate isaaclab_develop_6.0
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/tools/headless_trocar_stage_retry_demo.py \
  --model_path /localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/sft_4gpu_256bs_50ksteps_split_stage_prompt_free/checkpoint-50000 \
  --output_dir trocar_stage_retry_demo_prompt_free \
  --task1_max_steps 60 \
  --task2_max_steps 60 \
  --task_max_retries 3 \
  --perturb_tray_task 2 \
  --perturb_tray_after_steps 5 \
  --perturb_tray_duration_steps 5 \
  --initial_tray_yaw_deg 5 \
  --perturb_tray_yaw_deg 10 \
  --num_episodes 1 \
  --open_loop_steps 1 \
  --fps 15 \
  --fixed_initial_state_dataset /localhome/local-vennw/code/orca_trocar_data/assemble_trocar_sim_box_v3_60 \
  --fixed_initial_state_episode 0 \
  --fixed_initial_state_frame 0 \
  --fixed_initial_state_steps 30 \
  --fixed_initial_state_tolerance 0.035


  # involve cosmos
cd /localhome/local-vennw/code/cosmos-reason2
CUDA_VISIBLE_DEVICES=6 .venv/bin/vllm serve nvidia/Cosmos-Reason2-2B \
  --allowed-local-media-path /tmp/cosmos_reason2_queries \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --port 10086

# test cosmos

cd /localhome/local-vennw/code/IsaacLab
/localhome/local-vennw/miniconda3/envs/isaaclab_develop_6.0/bin/python \
  reason2_retry_hand_samples/eval_reason2_online.py --port 10086
