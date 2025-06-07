export PYTHONPATH=/localhome/local-vennw/code/Isaac-GR00T:$PYTHONPATH
python scripts/gr00t_finetune.py --dataset-path /localhome/local-vennw/code/Isaac-GR00T/so101_pick_and_place_home_fix_video \
--num-gpus 1 --batch-size 16 --max-steps 50000 \
--data_config so101_wrist --report_to tensorboard --video_backend torchvision_av \
--output_dir "so101_wrist_finetune_pick_and_place_home_fix_video"