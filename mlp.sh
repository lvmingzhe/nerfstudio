#!/bin/bash

# mlp_num_layers_values=("2" "8")
# mlp_hidden_units_values=("30" "70" "100") # low, medium, high
# embedding_dim_values=("30" "80" "120") # low, medium, high

mlp_num_layers_values=("2")
mlp_hidden_units_values=("70") # low, medium, high
embedding_dim_values=("80") # low, medium, high

for mlp_num_layers_val in ${mlp_num_layers_values[@]}
do
  for mlp_hidden_units_val in ${mlp_hidden_units_values[@]}
  do
    for embedding_dim_val in ${embedding_dim_values[@]}
    do
      echo "Running experiment with mlp-num-layers=$mlp_num_layers_val, mlp-hidden-units=$mlp_hidden_units_val and embedding-dim=$embedding_dim_val"

      experiment_name="experiment_mlp-num-layers_${mlp_num_layers_val}_mlp-hidden-units_${mlp_hidden_units_val}_embedding-dim_${embedding_dim_val}"

      CUDA_VISIBLE_DEVICES=2 ns-train nerfacto \
      --data data/kitti/seq05_100_1 \
      --vis wandb \
      --project-name kitti_0730_nerfacto \
      --pipeline.model.collider-params near_plane 2.0 far_plane 600.0 \
      --experiment-name $experiment_name \
      --pipeline.datamanager.camera-optimizer.mlp-hidden-units $mlp_hidden_units_val \
      --pipeline.datamanager.camera-optimizer.embedding-dim $embedding_dim_val \
      --pipeline.datamanager.camera-optimizer.mlp-num-layers $mlp_num_layers_val \
      --pipeline.datamanager.camera-optimizer.position_noise_std 0 \
      --pipeline.datamanager.camera-optimizer.orientation_noise_std 0.034
    done
  done
done
