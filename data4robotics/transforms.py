# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn
from torchvision import transforms


class _MediumAug(nn.Module):
    def __init__(self, pad=0, size=224):
        super().__init__()
        self.pad = pad
        self.size = size
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        extra_dim = len(x.shape) > 4
        if extra_dim:
            assert len(x.shape) == 5
            B, T, C, H, W = x.shape
            x = x.reshape((B * T, C, H, W))

        n, c, h, w = x.size()
        assert h == w
        if self.pad > 0:
            padding = tuple([self.pad] * 4)
            x = torch.nn.functional.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[: self.size]
        arange = arange.unsqueeze(0).repeat(self.size, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0,
            2 * self.pad + h - self.size + 1,
            size=(n, 1, 1, 2),
            device=x.device,
            dtype=x.dtype,
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = torch.nn.functional.grid_sample(
            x, grid, padding_mode="zeros", align_corners=False
        )
        x = self.norm(x)

        if extra_dim:
            return x.reshape((B, T, C, self.size, self.size))
        return x


def get_gpu_transform_by_name(name, size=224):
    if name == "gpu_medium":
        return _MediumAug(size=size)
    raise NotImplementedError


def get_transform_by_name(name, size=224):
    if "gpu" in name:
        return None

    if name == "preproc":
        return transforms.Compose(
            [
                transforms.Resize((size, size), antialias=False),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    if name == "basic":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size, scale=(0.2, 1.0), antialias=False
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    if name == "medium":
        kernel_size = int(0.05 * size)
        kernel_size = kernel_size + (1 - kernel_size % 2)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size, scale=(0.9, 1.0), antialias=False
                ),
                transforms.GaussianBlur(kernel_size=kernel_size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    if name == "hard":
        kernel_size = int(0.05 * size)
        kernel_size = kernel_size + (1 - kernel_size % 2)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size, scale=(0.2, 1.0), antialias=False
                ),
                transforms.GaussianBlur(kernel_size=kernel_size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    if name == "advanced":
        kernel_size = int(0.05 * size)
        kernel_size = kernel_size + (1 - kernel_size % 2)
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size, scale=(0.2, 1.0), antialias=False
                ),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=kernel_size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    raise NotImplementedError(f"{name} not found!")
