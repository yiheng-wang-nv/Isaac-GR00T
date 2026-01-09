python scripts/gr00t_finetune.py \
--dataset-path /localhome/local-vennw/code/dataset/install_trocar \
 --num-gpus 4 --batch-size 32 --output-dir g1_install_trocar_4gpu_32bs_30k_steps \
 --data-config unitree_g1 --video_backend decord --report_to tensorboard --max_steps 30000 --save-steps 10000

 python scripts/gr00t_finetune.py \
--dataset-path /localhome/local-vennw/code/datasets/install_trocar_from_tray_realsense_lerobot \
 --num-gpus 1 --batch-size 64 --output-dir g1_install_trocar_1gpu_64bs_50k_steps \
 --data-config unitree_g1 --video_backend decord --report_to tensorboard --max_steps 50000 --save-steps 10000

 python scripts/gr00t_finetune.py \
--dataset-path /localhome/local-vennw/code/datasets/install_trocar_from_tray_realsense_lerobot \
 --num-gpus 1 --batch-size 64 --output-dir g1_install_trocar_1gpu_64bs_50k_steps_53_data_finetune \
 --base_model_path /localhome/local-vennw/code/Isaac-GR00T/g1_install_trocar_1gpu_64bs_50k_steps \
 --data-config unitree_g1 --video_backend decord --report_to tensorboard --max_steps 10000 --save-steps 5000

 python scripts/gr00t_finetune.py \
--dataset-path /localhome/local-vennw/code/datasets/install_trocar_from_tray_realsense_lerobot \
 --num-gpus 1 --batch-size 64 --output-dir g1_install_trocar_1gpu_64bs_50k_steps_53_data \
 --data-config unitree_g1 --video_backend decord --report_to tensorboard --max_steps 50000 --save-steps 50000
