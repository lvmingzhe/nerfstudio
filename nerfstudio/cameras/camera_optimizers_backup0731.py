# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pose and Intrinsics Optimizers
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Union

import torch
import tyro
from jaxtyping import Float, Int
from torch import Tensor, nn
from typing_extensions import assert_never

from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, OptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
    SchedulerConfig,
)
from nerfstudio.utils import poses as pose_utils


def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    # print("v.shape ",v.shape)
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)

def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R

def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w

def sinusoidal_encoding(seq_len, dim):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
    pos_enc = torch.zeros(seq_len, dim)
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    
    if dim % 2 == 1:  # if dim is odd
        pos_enc[:, -1] = torch.sin(torch.squeeze(position) * torch.tensor(1.0))
    else:  # if dim is even
        pos_enc[:, 1::2] = torch.cos(position * div_term)
    return pos_enc

@dataclass
class CameraOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: CameraOptimizer)

    mode: Literal["off", "SO3xR3", "SE3", "MLP"] = "off"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""

    mlp_hidden_units: Optional[int] = 100
    mlp_num_layers: Optional[int] = 3
    embedding_dim: Optional[int] = 80 
    # num_cams: Optional[int] = None 


    position_noise_std: float = 0
    """Noise to add to initial positions. Useful for debugging."""

    orientation_noise_std: float = 0.01
    """Noise to add to initial orientations. Useful for debugging."""

    optimizer: OptimizerConfig = field(default_factory=lambda: AdamOptimizerConfig(lr=6e-4, eps=1e-15))
    """ADAM parameters for camera optimization."""

    scheduler: SchedulerConfig = field(default_factory=lambda: ExponentialDecaySchedulerConfig(max_steps=10000))
    """Learning rate scheduler for camera optimizer.."""

    param_group: tyro.conf.Suppress[str] = "camera_opt"
    """Name of the parameter group used for pose optimization. Can be any string that doesn't conflict with other
    groups."""


class CameraOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    config: CameraOptimizerConfig

    def __init__(
        self,
        config: CameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices

        # Initialize learnable parameters.
        if self.config.mode == "off":
            pass
        elif self.config.mode in ("SO3xR3", "SE3"):
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((self.num_cameras, 6), device=device))
        # elif self.config.mode == "MLP":
        #     self.embed = sinusoidal_encoding(self.num_cameras, self.config.embedding_dim).to(self.device)
            
        #     self.mlp_layers = []
        #     self.mlp_layers.append(nn.Linear(self.config.embedding_dim, self.config.mlp_hidden_units))
        #     self.mlp_layers.append(nn.ReLU())
        #     for _ in range(self.config.mlp_num_layers - 1):
        #         self.mlp_layers.append(nn.Linear(self.config.mlp_hidden_units, self.config.mlp_hidden_units))
        #         self.mlp_layers.append(nn.ReLU())
        #     self.mlp_layers.append(nn.Linear(self.config.mlp_hidden_units, 6))

        #     self.mlp = nn.Sequential(*self.mlp_layers)
        elif self.config.mode == "MLP":
            self.embed = sinusoidal_encoding(self.num_cameras, self.config.embedding_dim).to(self.device)
            
            self.mlp_layers_r = []
            self.mlp_layers_r.append(nn.Linear(self.config.embedding_dim, self.config.mlp_hidden_units))
            self.mlp_layers_r.append(nn.ReLU())
            for _ in range(self.config.mlp_num_layers - 1):
                self.mlp_layers_r.append(nn.Linear(self.config.mlp_hidden_units, self.config.mlp_hidden_units))
                self.mlp_layers_r.append(nn.ReLU())
            self.mlp_layers_r.append(nn.Linear(self.config.mlp_hidden_units, 3))
            self.mlp_r = nn.Sequential(*self.mlp_layers_r)

            self.mlp_layers_t = []
            self.mlp_layers_t.append(nn.Linear(self.config.embedding_dim, self.config.mlp_hidden_units))
            self.mlp_layers_t.append(nn.ReLU())
            for _ in range(self.config.mlp_num_layers - 1):
                self.mlp_layers_t.append(nn.Linear(self.config.mlp_hidden_units, self.config.mlp_hidden_units))
                self.mlp_layers_t.append(nn.ReLU())
            self.mlp_layers_t.append(nn.Linear(self.config.mlp_hidden_units, 3))
            self.mlp_t = nn.Sequential(*self.mlp_layers_t)
        else:
            assert_never(self.config.mode)

        # Initialize pose noise; useful for debugging.
        if config.position_noise_std != 0.0 or config.orientation_noise_std != 0.0:
            assert config.position_noise_std >= 0.0 and config.orientation_noise_std >= 0.0
            std_vector = torch.tensor(
                [config.position_noise_std] * 3 + [config.orientation_noise_std] * 3, device=device
            )
            self.pose_noise = exp_map_SE3(torch.normal(torch.zeros((self.num_cameras, 6), device=device), std_vector))
        else:
            self.pose_noise = None

    def forward(
        self,
        indices: Int[Tensor, "num_cameras"],
    ) -> Float[Tensor, "num_cameras 3 4"]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = []

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            outputs.append(exp_map_SO3xR3(self.pose_adjustment[indices, :]))
        elif self.config.mode == "SE3":
            outputs.append(exp_map_SE3(self.pose_adjustment[indices, :]))
        # elif self.config.mode == "MLP":
        #     # Indices might need to be reshaped, normalized or otherwise preprocessed
        #     x = self.embed[indices]  # (num_cams, embedding_dim)
        #     tangent_vector = self.mlp(x) # Output (num_cams, 6)
        #     outputs.append(exp_map_SO3xR3(tangent_vector))   
        elif self.config.mode == "MLP":
            # Indices might need to be reshaped, normalized or otherwise preprocessed
            x = self.embed[indices]  # (num_cams, embedding_dim)
            r = self.mlp_r(x)  # Output (num_cams, 3)
             # If translation optimization is not desired, produce a zero vector
            optimize_t = False
            if optimize_t == False:
                t = torch.zeros_like(r).to(x.device) # Creates a zero tensor of the same size and type as r, and moves it to the appropriate device
            else:
                t = self.mlp_t(x)
            # t = self.mlp_t(x)  # Output (num_cams, 3)
            tangent_vector = torch.cat((r, t), dim=-1)  # Concatenating rotation and translation (num_cams, 6)
            outputs.append(exp_map_SO3xR3(tangent_vector))  
        else:
            assert_never(self.config.mode)
        # Detach non-trainable indices by setting to identity transform
        if self.non_trainable_camera_indices is not None:
            outputs[0][self.non_trainable_camera_indices] = torch.eye(4, device=self.device)[:3, :4]

        # Apply initial pose noise.
        if self.pose_noise is not None:
            outputs.append(self.pose_noise[indices, :, :])
        # Return: identity if no transforms are needed, otherwise multiply transforms together.
        if len(outputs) == 0:
            # Note that using repeat() instead of tile() here would result in unnecessary copies.
            return torch.eye(4, device=self.device)[None, :3, :4].tile(indices.shape[0], 1, 1)
        return functools.reduce(pose_utils.multiply, outputs)
