"""
Filename: hermes/revalexo/ai/utils/transforms.py
Author: Diwas Lamsal <diwaslamsal123@hotmail.com>
Date: 2026-03-06
Version: 1.0
Description: Input data transform builders for live inference.
    Wraps torchvision and pytorchvideo transforms for video, image, and IMU modalities.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable


class DivideBy255:
    """Divide tensor values by 255 to normalize to [0, 1]."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255.0


class NormalizeVideo:
    """Normalize video tensor with mean and std per channel.

    Expects input [C, T, H, W].
    """

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean).view(-1, 1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        return (x - self.mean) / self.std


class ShortSideScale:
    """Scale the shorter side of video frames to a target size.

    Expects input [C, T, H, W].
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C, T, H, W]
        c, t, h, w = x.shape
        if h < w:
            new_h = self.size
            new_w = int(w * self.size / h)
        else:
            new_w = self.size
            new_h = int(h * self.size / w)

        # Reshape to [C*T, 1, H, W] for interpolation then back
        x = x.permute(0, 1, 2, 3)  # [C, T, H, W]
        x = x.reshape(c * t, 1, h, w)
        x = nn.functional.interpolate(
            x, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        x = x.reshape(c, t, new_h, new_w)
        return x


class CenterCropVideo:
    """Center crop video frames.

    Expects input [C, T, H, W].
    """

    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = tuple(crop_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C, T, H, W]
        _, _, h, w = x.shape
        crop_h, crop_w = self.crop_size
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        return x[:, :, start_h : start_h + crop_h, start_w : start_w + crop_w]


class NormalizeImage:
    """Normalize image tensor with ImageNet mean and std.

    Expects input [1, C, H, W] with values in [0, 255].
    """

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        # Divide by 255 first if values are in [0, 255]
        if x.max() > 1.0:
            x = x / 255.0
        return (x - self.mean) / self.std


class ResizeImage:
    """Resize image shorter side to target size.

    Expects input [1, C, H, W].
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, C, H, W]
        _, _, h, w = x.shape
        if h < w:
            new_h = self.size
            new_w = int(w * self.size / h)
        else:
            new_w = self.size
            new_h = int(h * self.size / w)

        x = nn.functional.interpolate(
            x, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        return x  # [1, C, H, W]


class CenterCropImage:
    """Center crop image.

    Expects input [1, C, H, W].
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        start_h = (h - self.size) // 2
        start_w = (w - self.size) // 2
        return x[:, :, start_h : start_h + self.size, start_w : start_w + self.size]


class CenterSquareCrop:
    """Center-crop the largest possible square (side = min(H, W)).

    Expects input [1, C, H, W]. Use before ResizeImage to preserve a wider
    field of view than the shorter-side-resize + center-crop chain.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        side = min(h, w)
        start_h = (h - side) // 2
        start_w = (w - side) // 2
        return x[:, :, start_h : start_h + side, start_w : start_w + side]


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# Registry mapping transform names to classes
TRANSFORM_REGISTRY = {
    "DivideBy255": DivideBy255,
    "NormalizeVideo": NormalizeVideo,
    "ShortSideScale": ShortSideScale,
    "CenterCropVideo": CenterCropVideo,
    "NormalizeImage": NormalizeImage,
    "ResizeImage": ResizeImage,
    "CenterCropImage": CenterCropImage,
    "CenterSquareCrop": CenterSquareCrop,
}


def build_transforms(transform_configs: List[Dict[str, Any]]) -> Optional[Compose]:
    """Build a transform pipeline from config specifications.

    Args:
        transform_configs: List of dicts with 'name' and optional 'params'.

    Returns:
        Optional[Compose] Composed transform pipeline or `None` if empty.
    """
    if not transform_configs:
        return None

    transforms = []
    for t_config in transform_configs:
        name = t_config.get("name")
        params = t_config.get("params", {})

        if name in TRANSFORM_REGISTRY:
            transforms.append(TRANSFORM_REGISTRY[name](**params))
        else:
            print(f"Warning: Unknown transform '{name}', skipping.")

    if transforms:
        return Compose(transforms)
    return None
