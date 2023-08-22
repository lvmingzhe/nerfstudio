#!/bin/bash

# ns-process-data images --data data/nerfstudio/llff_flower/ --output-dir data/nerfstudio/llff_flower_processed

# mode_val="MLP"
# mode_val=("SO3xR3")
mode_val=("off")
experiment_name="experiment_flower_${mode_val}"

# CUDA_VISIBLE_DEVICES=2 ns-train nerfacto \
CUDA_VISIBLE_DEVICES=2 ns-train instant-ngp \
    --data data/nerfstudio/llff_flower_processed \
    --vis viewer+wandb \
    --project-name flower_0802_nerfacto \
    --pipeline.model.collider-params near_plane 1 far_plane 1000.0 \
    --experiment-name $experiment_name \
    --pipeline.datamanager.camera-optimizer.mode $mode_val \
    --pipeline.datamanager.camera-optimizer.position_noise_std 0 \
    --pipeline.datamanager.camera-optimizer.orientation_noise_std 0.0