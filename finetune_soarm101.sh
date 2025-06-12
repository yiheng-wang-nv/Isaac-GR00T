python scripts/gr00t_finetune.py \
   --dataset-path /localhome/local-vennw/code/Isaac-GR00T/finetune_data/so101_scissors_2_cameras \
   --num-gpus 1 \
   --batch-size 64 \
   --output-dir so101_scissors_2_cameras_finetune \
   --max-steps 10000 \
   --data-config so100_dualcam \
   --report_to tensorboard \
   --video-backend torchvision_av