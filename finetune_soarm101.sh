export PYTHONPATH=/localhome/local-vennw/code/Isaac-GR00T:$PYTHONPATH
python scripts/gr00t_finetune.py --dataset-path /localhome/local-vennw/.cache/huggingface/lerobot/Venn/so101_pick_and_place \
--num-gpus 1 --batch-size 16 --max-steps 10000 \
--data_config so101_wrist --report_to tensorboard --video_backend torchvision_av
--output_dir "so101_wrist_finetune_pick_and_place"