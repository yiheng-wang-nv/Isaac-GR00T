# gr00t n1
python scripts/gr00t_n1_finetune.py \
   --dataset-path /localhome/local-vennw/code/Isaac-GR00T/finetune_data/so101_scissors_2_cameras_fps15  \
   --num-gpus 1 \
   --batch-size 64 \
   --output-dir so101_scissors_2_cameras_fps15_finetune_n1 \
   --max-steps 10000 \
   --data-config so100_dualcam \
   --report_to tensorboard \
   --video-backend torchvision_av

# gr00t n1.5
python scripts/gr00t_finetune.py \
   --dataset-path /localhome/local-vennw/code/Isaac-GR00T/finetune_data/so101_scissors_2_cameras_fps15 \
   --num-gpus 1 \
   --batch-size 32 \
   --output-dir so101_scissors_2_cameras_fps15_finetune \
   --max-steps 10000 \
   --data-config so100_dualcam \
   --report_to tensorboard \
   --base_model_path /localhome/local-vennw/code/Isaac-GR00T/so101_scissors_2_cameras_finetune/checkpoint-8000 \
   --video-backend torchvision_av

# check finetune for checkpoint-3000 to 10000
for i in {10000..10000..1000}; do
   python scripts/eval_policy.py --plot \
      --embodiment_tag new_embodiment \
      --model_path so101_scissors_2_cameras_fps15_finetune/checkpoint-${i} \
      --data_config so100_dualcam \
      --dataset_path finetune_data/so101_scissors_2_cameras_fps15 \
      --video_backend torchvision_av \
      --modality_keys single_arm gripper
   echo "Finished checkpoint-${i}"
done

# deploy
python scripts/inference_service.py --server \
    --model_path so101_scissors_2_cameras_fps15_finetune/checkpoint-8000 \
    --embodiment-tag new_embodiment \
    --data-config so100_dualcam \
    --denoising-steps 4

python getting_started/examples/eval_lerobot.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=so101_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, room: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --policy_host=localhost \
    --lang_instruction="Grip a straight scissor and put it in the box."

# "Grip a tweezer and put it in the box."
# "Grip a straight scissor and put it in the box."
