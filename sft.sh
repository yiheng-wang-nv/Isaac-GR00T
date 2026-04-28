# Navigate to the GR00T repository
# If you are using the rheo workflow container with `-g1.5` flag, the path is `third_party/Isaac-GR00T`

source "/localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/third_party/cosmos-transfer2.5/.venv/bin/activate"

# full data base model
CUDA_VISIBLE_DEVICES=4,5,6,7 IS_TORCHRUN=1 \
torchrun --standalone --nproc_per_node=4 --nnodes=1 scripts/gr00t_finetune.py \
  --dataset-path /localhome/local-vennw/data/trocar_parallel_combined_success \
  --num-gpus 4 \
  --batch-size 64 \
  --output-dir sft_4gpu_512bs_50ksteps \
  --data-config gr00t_config:UnitreeG1SimNoStageDataConfig \
  --video_backend decord \
  --report_to tensorboard \
  --max_steps 50000 \
  --save-steps 10000 --gradient_accumulation_steps 2

# full data base model tune visual
CUDA_VISIBLE_DEVICES=0,1 IS_TORCHRUN=1 \
torchrun --standalone --nproc_per_node=2 --nnodes=1 scripts/gr00t_finetune.py \
  --dataset-path /localhome/local-vennw/data/trocar_parallel_combined_success \
  --num-gpus 2 \
  --batch-size 24 \
  --output-dir sft_2gpu_256bs_50ksteps_tune_visual \
  --data-config gr00t_config:UnitreeG1SimNoStageDataConfig \
  --video_backend decord \
  --report_to tensorboard \
  --max_steps 50000 \
  --save-steps 10000 \
  --tune_visual

# split data + prompt-free stage classifier
CUDA_VISIBLE_DEVICES=4,5,6,7 IS_TORCHRUN=1 \
torchrun --standalone --nproc_per_node=4 --nnodes=1 scripts/gr00t_finetune.py \
  --dataset-path /localhome/local-vennw/data/trocar_parallel_combined_split \
  --num-gpus 4 \
  --batch-size 64 \
  --output-dir sft_4gpu_256bs_50ksteps_split_stage_prompt_free \
  --data-config gr00t_config:UnitreeG1SimDataConfig \
  --video_backend decord \
  --report_to tensorboard \
  --max_steps 50000 \
  --save-steps 25000 \
  --use-stage-classifier

# full data + background augmentation (success dataset, no stage column → disable stage)
# Templates pulled from HealthSurgiBench_1e4/media via DEFAULT_TEMPLATE_FOLDER in gr00t_mask_config.py
CUDA_VISIBLE_DEVICES=0,1 IS_TORCHRUN=1 \
torchrun --standalone --nproc_per_node=2 --nnodes=1 scripts/gr00t_finetune.py \
  --dataset-path /localhome/local-vennw/data/trocar_parallel_combined_success \
  --num-gpus 2 \
  --batch-size 64 \
  --output-dir sft_2gpu_256bs_50ksteps_bgaug \
  --data-config gr00t_mask_config:UnitreeG1SimMaskDataConfig \
  --video_backend decord \
  --report_to tensorboard \
  --max_steps 50000 \
  --save-steps 10000 \
  --no-use-stage-classifier

# dry-run: stage classifier sanity check (20 steps, GPU 2 only)
CUDA_VISIBLE_DEVICES=2 python scripts/gr00t_finetune.py \
  --dataset-path /localhome/local-vennw/data/trocar_parallel_combined_split \
  --num-gpus 1 \
  --batch-size 4 \
  --output-dir /tmp/stage_dry_run \
  --data-config gr00t_config:UnitreeG1SimDataConfig \
  --video_backend decord \
  --report_to tensorboard \
  --max_steps 20 \
  --save-steps 20 \
  --use-stage-classifier

# split data + stage classifier only (no background augmentation)
CUDA_VISIBLE_DEVICES=2,3,4,5 IS_TORCHRUN=1 \
torchrun --standalone --nproc_per_node=4 --nnodes=1 scripts/gr00t_finetune.py \
  --dataset-path /localhome/local-vennw/data/trocar_parallel_combined_split \
  --num-gpus 4 \
  --batch-size 64 \
  --output-dir sft_2gpu_256bs_80ksteps_split_stage \
  --data-config gr00t_config:UnitreeG1SimDataConfig \
  --video_backend decord \
  --report_to tensorboard \
  --max_steps 80000 \
  --save-steps 10000 \
  --use-stage-classifier

# eval: SFT model, 100 episodes, 512 steps, no mask
conda activate isaaclab_develop_6.0
CUDA_VISIBLE_DEVICES=1 python /localhome/local-vennw/code/IsaacLab/scripts/tools/record_trocar_episodes.py \
  --model_path /localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/sft_2gpu_256bs_50ksteps/checkpoint-50000 \
  --output_dir /localhome/local-vennw/code/sft_eval_100_8_steps_rand_light \
  --num_episodes 100 \
  --max_steps 512 \
  --use_gr00t_policy \
  --open_loop_steps 8 \
  --fixed_initial_state_dataset /localhome/local-vennw/code/orca_trocar_data/assemble_trocar_sim_box_v3_60 \
  --fixed_initial_state_episode 0 \
  --fixed_initial_state_frame 0 \
  --fixed_initial_state_steps 30 \
  --fixed_initial_state_tolerance 0.035 --randomize_lighting --seed 42

CUDA_VISIBLE_DEVICES=3 python /localhome/local-vennw/code/IsaacLab/scripts/tools/record_trocar_episodes.py \
  --model_path /localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/sft_2gpu_256bs_50ksteps/checkpoint-50000 \
  --output_dir /localhome/local-vennw/code/sft_eval_100_4_steps \
  --num_episodes 100 \
  --max_steps 512 \
  --use_gr00t_policy \
  --open_loop_steps 4 \
  --fixed_initial_state_dataset /localhome/local-vennw/code/orca_trocar_data/assemble_trocar_sim_box_v3_60 \
  --fixed_initial_state_episode 0 \
  --fixed_initial_state_frame 0 \
  --fixed_initial_state_steps 30 \
  --fixed_initial_state_tolerance 0.035

CUDA_VISIBLE_DEVICES=3 python /localhome/local-vennw/code/IsaacLab/scripts/tools/record_trocar_episodes.py \
  --model_path /localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/sft_4gpu_512bs_50ksteps/checkpoint-50000 \
  --output_dir /localhome/local-vennw/code/sft_eval_100_large_bs_8_steps \
  --num_episodes 100 \
  --max_steps 512 \
  --use_gr00t_policy \
  --open_loop_steps 8 \
  --fixed_initial_state_dataset /localhome/local-vennw/code/orca_trocar_data/assemble_trocar_sim_box_v3_60 \
  --fixed_initial_state_episode 0 \
  --fixed_initial_state_frame 0 \
  --fixed_initial_state_steps 30 \
  --fixed_initial_state_tolerance 0.035

CUDA_VISIBLE_DEVICES=6 python /localhome/local-vennw/code/IsaacLab/scripts/tools/record_trocar_episodes.py \
  --model_path /localhome/local-vennw/code/ORCA-Assemble-Trocar-GR00T-RL-Dev/g1_install_trocar_sim_box_v3_60_train_bs32_1_gpus_cos_30k_tune_visual \
  --output_dir /localhome/local-vennw/data/sft_eval_yun_100 \
  --num_episodes 100 \
  --max_steps 512 \
  --use_gr00t_policy \
  --no_mask

# check multi-stage
cd /localhome/local-vennw/code/IsaacLab
conda activate isaaclab_develop_6.0
DISPLAY=:1 XAUTHORITY=$HOME/.Xauthority \
  ./isaaclab.sh -p scripts/tools/interactive_trocar_multitask.py \
    --model_path /localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/sft_2gpu_256bs_80ksteps_split_stage/checkpoint-80000 \
    --device cuda:0 \
    --viz kit

# check demo
cd /localhome/local-vennw/code/IsaacLab
conda activate isaaclab_develop_6.0
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/tools/headless_trocar_stage_retry_demo.py \
  --model_path /localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/sft_2gpu_256bs_80ksteps_split_stage/checkpoint-80000 \
  --output_dir trocar_stage_retry_demo \
  --task_max_steps 60 \
  --task3_max_retries 3 \
  --task4_max_steps 60 \
  --task4_max_retries 3 \
  --back_to_init_steps 60 \
  --back_to_init_tolerance 0.035 \
  --num_episodes 5 \
  --fps 30