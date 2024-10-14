# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math

import numpy as np
import torch
import torch.nn as nn
from r3m import load_r3m
from torchvision import models

from data4robotics.models.base import BaseModel


def _make_norm(norm_cfg):
    if norm_cfg["name"] == "batch_norm":
        return nn.BatchNorm2d
    if norm_cfg["name"] == "group_norm":
        num_groups = norm_cfg["num_groups"]
        return lambda num_channels: nn.GroupNorm(num_groups, num_channels)
    if norm_cfg["name"] == "diffusion_policy":

        def _gn_builder(num_channels):
            num_groups = int(num_channels // 16)
            return nn.GroupNorm(num_groups, num_channels)

        return _gn_builder
    raise NotImplementedError(f"Missing norm layer: {norm_cfg['name']}")


def _construct_resnet(size, norm, weights=None):
    if size == 18:
        w = models.ResNet18_Weights
        m = models.resnet18(norm_layer=norm)
    elif size == 34:
        w = models.ResNet34_Weights
        m = models.resnet34(norm_layer=norm)
    elif size == 50:
        w = models.ResNet50_Weights
        m = models.resnet50(norm_layer=norm)
    else:
        raise NotImplementedError(f"Missing size: {size}")

    if weights is not None:
        w = w.verify(weights).get_state_dict(progress=True)
        if norm is not nn.BatchNorm2d:
            w = {
                k: v
                for k, v in w.items()
                if "running_mean" not in k and "running_var" not in k
            }
        m.load_state_dict(w)
    return m


class ResNet(BaseModel):
    def __init__(
        self,
        size,
        norm_cfg,
        weights=None,
        restore_path="",
        avg_pool=True,
        conv_repeat=0,
    ):
        norm_layer = _make_norm(norm_cfg)
        model = _construct_resnet(size, norm_layer, weights)
        model.fc = nn.Identity()
        if not avg_pool:
            model.avgpool = nn.Identity()

        if conv_repeat > 1:
            w = model.conv1.weight.data.repeat((1, conv_repeat, 1, 1))
            model.conv1.weight.data = w
            model.conv1.in_channels *= conv_repeat

        super().__init__(model, restore_path)
        self._size, self._avg_pool = size, avg_pool

    def forward(self, x):
        if self._avg_pool:
            return self._model(x)[:, None]
        B = x.shape[0]
        x = self._model(x)
        x = x.reshape((B, self.embed_dim, -1))
        return x.transpose(1, 2)

    @property
    def embed_dim(self):
        return {18: 512, 34: 512, 50: 2048}[self._size]

    @property
    def n_tokens(self):
        if self._avg_pool:
            return 1
        return 49  # assuming 224x224 images


class R3M(ResNet):
    def __init__(self, size, avg_pool=True):
        nn.Module.__init__(self)
        self._model = load_r3m(f"resnet{size}").module.convnet.cpu()
        if not avg_pool:
            self._model.avgpool = nn.Identity()
        self._size, self._avg_pool = size, avg_pool


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.
    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """

    def __init__(
        self,
        input_shape,
        num_kp=None,
        temperature=1.0,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not use spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution,
            and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape  # (C, H, W)

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter("temperature", temperature)
        else:
            # temperature held constant after initialization
            temperature = nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer("temperature", temperature)

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert len(input_shape) == 3
        assert input_shape[0] == self._in_c
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial
        probability distribution is created using a softmax, where the support is the
        pixel locations. This distribution is used to compute the expected value of
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.
        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert feature.shape[1] == self._in_c
        assert feature.shape[2] == self._in_h
        assert feature.shape[3] == self._in_w
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = torch.nn.functional.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1]
        # for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution
            # and compute second-order statistics to return
            expected_xx = torch.sum(
                self.pos_x * self.pos_x * attention, dim=1, keepdim=True
            )
            expected_yy = torch.sum(
                self.pos_y * self.pos_y * attention, dim=1, keepdim=True
            )
            expected_xy = torch.sum(
                self.pos_x * self.pos_y * attention, dim=1, keepdim=True
            )
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2]
            # where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(
                -1, self._num_kp, 2, 2
            )
            feature_keypoints = (feature_keypoints, feature_covar)

        return feature_keypoints


class RobomimicResNet(nn.Module):
    def __init__(self, size, norm_cfg, weights=None, img_size=224, feature_dim=64):
        super().__init__()
        norm_layer = _make_norm(norm_cfg)
        model = _construct_resnet(size, norm_layer, weights)
        # Cut the last two layers.
        self.resnet = nn.Sequential(*(list(model.children())[:-2]))
        resnet_out_dim = int(math.ceil(img_size / 32.0))
        resnet_output_shape = [512, resnet_out_dim, resnet_out_dim]
        self.spatial_softmax = SpatialSoftmax(
            resnet_output_shape,
            num_kp=64,
            temperature=1.0,
            noise_std=0.0,
            output_variance=False,
            learnable_temperature=False,
        )
        pool_output_shape = self.spatial_softmax.output_shape(resnet_output_shape)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.proj = nn.Linear(int(np.prod(pool_output_shape)), feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.resnet(x)
        x = self.spatial_softmax(x)
        x = self.flatten(x)
        x = self.proj(x)
        return x[:, None]

    @property
    def embed_dim(self):
        return self.feature_dim

    @property
    def n_tokens(self):
        return 1
