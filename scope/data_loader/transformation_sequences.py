import numpy as np
from torchvision import transforms

from data_loader.transformations import RandomFlipTransform, RandomAffineTransform, RandomNoiseTransform, \
    RandomElasticTransform, ToFloat, ToTensor, ChannelsFirst


def gsd_pCT_train_transform(max_output_channels=2, random_flip_prob=0.5, random_elastic_prob=0.5, random_affine_prob=0.5,
                            random_noise_prob=0.5, shift_val = (0, 5), rotate_val = 15.0, scale_val = (0.7, 1.3), flip_axis = (0),
                            flip_prob_per_axis = 0.5, noise_std = (0, 0.25), noise_mean = 0, max_deform = (7.5, 7.5, 7.5),
                            elastic_control_points = (7, 7, 7), seed=None, prudent=True, verbose=False):
    if seed is None:
        seed = np.random.randint(0, 9999)  # seed must be an integer for torch

    train_transform = transforms.Compose([
        # ToTensor(),
        # ToFloat(),
        RandomFlipTransform(axes=flip_axis, flip_probability=flip_prob_per_axis, p=random_flip_prob,
                            seed=seed, max_output_channels=max_output_channels, prudent=prudent),
        RandomElasticTransform(max_displacement=max_deform,
                               num_control_points=elastic_control_points,
                               image_interpolation='bspline',
                               seed=seed, p=random_elastic_prob,
                               max_output_channels=max_output_channels, verbose=verbose, prudent=prudent),
        RandomAffineTransform(scales=scale_val, degrees=rotate_val, translation=shift_val,
                              isotropic=True, default_pad_value=0,
                              image_interpolation='bspline', seed=seed, p=random_affine_prob,
                              max_output_channels=max_output_channels, verbose=verbose, prudent=prudent),
        RandomNoiseTransform(mean=noise_mean, std=noise_std, seed=seed, p=random_noise_prob,
                             max_output_channels=max_output_channels, prudent=prudent),
        # ChannelsFirst(),
        # ToFloat(),
    ])

    return train_transform


def gsd_pCT_valid_transform(seed=None):
    valid_transform = transforms.Compose([
        ToTensor(),
        ToFloat(),
        ChannelsFirst(),
        ToFloat(),
    ])

    return valid_transform