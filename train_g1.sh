python scripts/gr00t_finetune.py \
--dataset-path /localhome/local-vennw/code/dataset/install_trocar \
 --num-gpus 4 --batch-size 32 --output-dir g1_install_trocar_4gpu_32bs_30k_steps \
 --data-config unitree_g1 --video_backend decord --report_to tensorboard --max_steps 30000 --save-steps 10000
