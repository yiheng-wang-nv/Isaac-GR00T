# gr00t n1.5 finetune
python scripts/gr00t_finetune.py \
   --dataset-path /localhome/local-vennw/code/Isaac-GR00T/finetune_data/2_cameras_fps15_enhanced_gripper \
   --num-gpus 1 \
   --batch-size 32 \
   --output-dir 2_cameras_fps15_enhanced_gripper_finetune_5k_196_data \
   --max-steps 5000 \
   --save-steps 1000 \
   --data-config so100_dualcam \
   --report_to tensorboard \
   --video-backend torchvision_av

# add more data further finetune
python scripts/gr00t_finetune.py \
   --dataset-path /localhome/local-vennw/code/Isaac-GR00T/finetune_data/2_cameras_fps15_enhanced_gripper \
   --num-gpus 1 \
   --batch-size 32 \
   --output-dir 2_cameras_fps15_enhanced_gripper_finetune_30k_196_data \
   --max-steps 3000 \
   --save-steps 1000 \
   --data-config so100_dualcam \
   --report_to tensorboard \
   --video-backend torchvision_av \
   --base_model_path /localhome/local-vennw/code/Isaac-GR00T/2_cameras_fps15_enhanced_gripper_finetune_30k_196_data/checkpoint-30000

# 231 data finetune
python scripts/gr00t_finetune.py \
   --dataset-path /localhome/local-vennw/code/Isaac-GR00T/finetune_data/2_cameras_fps15_enhanced_gripper \
   --num-gpus 1 \
   --batch-size 32 \
   --output-dir 2_cameras_fps15_enhanced_gripper_finetune_30k_231_data \
   --max-steps 30000 \
   --save-steps 5000 \
   --data-config so100_dualcam \
   --report_to tensorboard \
   --video-backend torchvision_av

# deploy
python scripts/inference_service.py --server \
    --model_path 2_cameras_fps15_enhanced_gripper_finetune_30k_231_data/checkpoint-30000 \
    --embodiment-tag new_embodiment \
    --data-config so100_dualcam \
    --denoising-steps 4

# inference request, scissor
# sudo chmod 666 /dev/ttyACM1

python eval_gr00t_so101.py \
   --host 127.0.0.1 \
   --port 5555 \
   --port_follower /dev/ttyACM2 \
   --task_description "Grip a straight scissor and put it in the box." \
   --actions_to_execute 30 \
   --record_images \
   --create_videos \
   --output_dir scissor_inference_output_single

# inference request, tweezer
python eval_gr00t_so101.py \
   --host 127.0.0.1 \
   --port 5555 \
   --port_follower /dev/ttyACM2 \
   --task_description "Grip a tweezer and put it in the box." \
   --actions_to_execute 30 \
   --record_images \
   --create_videos \
   --output_dir tweezer_inference_output_single
