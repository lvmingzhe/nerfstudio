#!/bin/bash

# std_values=("0.034" "0.017")
# mode_values=("off" "SO3xR3" "SE3" "MLP")

# for std_val in ${std_values[@]}
# do
#   for mode_val in ${mode_values[@]}
#   do
#     echo "Running experiment with orientation_noise_std=$std_val and mode=$mode_val"

#     experiment_name="experiment_std_${std_val}_mode_${mode_val}"
    
#     CUDA_VISIBLE_DEVICES=2 ns-train nerfacto \
#     --data data/kitti/seq05_100_1 \
#     --vis wandb \
#     --project-name kitti_0730_nerfacto \
#     --pipeline.model.collider-params near_plane 2.0 far_plane 600.0 \
#     --experiment-name $experiment_name \
#     --pipeline.datamanager.camera-optimizer.mode $mode_val \
#     --pipeline.datamanager.camera-optimizer.position_noise_std 0 \
#     --pipeline.datamanager.camera-optimizer.orientation_noise_std $std_val
#   done
# done

# mode_val="MLP"
# mode_val=("SO3xR3")
# mode_val=("off")
# experiment_name="experiment_std_0034_mode_${mode_val}"

# CUDA_VISIBLE_DEVICES=2 ns-train instant-ngp \

# CUDA_VISIBLE_DEVICES=1 ns-train nerfacto \
#     --data data/kitti/seq05_100_1 \
#     --vis wandb \
#     --project-name kitti_0803_nerfacto \
#     --pipeline.model.collider-params near_plane 2.0 far_plane 600.0 \
#     --experiment-name $experiment_name \
#     --pipeline.datamanager.camera-optimizer.mode $mode_val \
#     --pipeline.datamanager.camera-optimizer.position_noise_std 0.0 \
#     --pipeline.datamanager.camera-optimizer.orientation_noise_std 0.0