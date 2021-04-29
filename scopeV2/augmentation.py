import math
import random

import torch.nn.functional as F
import torch
from torch import nn as nn


class Augmentation(nn.Module):
    # gpu based augmentation module
    def __init__(
            self, flip=None, offset=None, scale=None, rotate=None, noise=None
    ):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, image_tensor: torch.Tensor):
        """
        Apply random augmentation of flip/offset/rotate/scale/noise
        image_tensor: tensor with dimensions (C,H,W,D)

        returns: tensor with dimensions (C,H,W,D)
        """

        image_tensor = image_tensor.unsqueeze(0)
        transform_t = self._build3DTransformMatrix()
        transform_t = transform_t.to(image_tensor.device, torch.float32)

        affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0),
            image_tensor.size(),
            align_corners=False,
        )

        augmented_image = F.grid_sample(
            image_tensor,
            affine_t,
            padding_mode='border',
            align_corners=False,
        )

        if self.noise:
            noise_t = torch.randn_like(augmented_image)
            noise_t *= self.noise

            augmented_image += noise_t

        return augmented_image[0]


    def _build3DTransformMatrix(self):
        transform_t = torch.eye(4)

        # along every axis (except channel)
        for i in range(3):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i, i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transform_t[i, 3] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transform_t[i, i] *= 1.0 + scale_float * random_float

        if self.rotate:
            # only rotate along x/y axes as voxel-spacing along z is not the same
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])

            transform_t @= rotation_t
        return transform_t


def apply_augmentation(image_tensor: torch.Tensor, augmentation_dict: dict):
    """
    Apply random augmentation of flip/offset/rotate/scale/noise (CPU implementation)
    image_tensor: tensor with dimensions (C,H,W,D)

    returns: tensor with dimensions (C,H,W,D)
    """

    # adding a sample dimension as grid sample expects input in the form of (N,C,H,W,D)
    image_tensor = image_tensor.unsqueeze(0)
    transform_t = torch.eye(4)

    # along every axis (except channel)
    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i, i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i, 3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i, i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:
        # only rotate along x/y axes as voxel-spacing along z is not the same
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        transform_t @= rotation_t

    affine_t = F.affine_grid(
        transform_t[:3].unsqueeze(0).to(torch.float32),
        image_tensor.size(),
        align_corners=False,
    )

    augmented_image = F.grid_sample(
        image_tensor,
        affine_t,
        padding_mode='border',
        align_corners=False,
    ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_image)
        noise_t *= augmentation_dict['noise']

        augmented_image += noise_t

    return augmented_image[0]


