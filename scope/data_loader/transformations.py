import numpy as np
import collections
from typing import Optional, Tuple, Union

import torch
from torchio.transforms import RandomAffine, RandomFlip, RandomNoise, RandomElasticDeformation


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, input):
        return torch.from_numpy(input)


class ChannelsFirst(object):
    """Convert to channels First."""

    def __call__(self, input):
        return input.permute(3, 0, 1, 2)


class ToFloat(object):
    """Convert to float tensor."""

    def __call__(self, input):
        return input.to(torch.float32)


class TorchIOTransformer(object):
    def __init__(self, get_transformer, max_output_channels=10, prudent=True, verbose=False):
        self.get_transformer = get_transformer
        self.max_output_channels = max_output_channels
        self.prudent = prudent
        self.verbose = verbose

    def __call__(self, *inputs):
        if not (isinstance(inputs, collections.Sequence) or isinstance(inputs, np.ndarray)):
            inputs = [inputs]

        outputs = []
        for idx, _input in enumerate(inputs):
            # _input = _input.permute(3, 0, 1, 2)  # channels first for torchio
            # Detect masks (label mask and brain mask)
            n_unique = list(_input.unique().size())[0]
            if n_unique <= self.max_output_channels or n_unique <= 2:
                transformer = self.get_transformer(mask=True)
                input_tf = transformer(_input)
                input_tf = input_tf.round()
                if _input.unique().size() != input_tf.unique().size():
                    if self.verbose:
                        print(f'WARNING... Input mask and its transformation differ in number of classes: '
                              f'input {_input.unique().size()} vs. transformed {input_tf.unique().size()} '
                              f'for {transformer} and number of voxels in initial mask: {_input.sum()}')
                    if self.prudent:
                        if self.verbose: print('Returning non transformed input.')
                        # Avoid loss of classes by transformation
                        # (either due to extreme transformation or very little voxels of a certain class present)
                        return inputs  # return bot all inputs untransformed
            else:
                transformer = self.get_transformer()
                input_tf = transformer(_input)
            # input_tf = input_tf.permute(1, 2, 3, 0)  # replace channels last

            outputs.append(input_tf)

        return outputs if idx >= 1 else outputs[0]


class RandomElasticTransform(TorchIOTransformer):
    def __init__(
            self,
            num_control_points: Union[int, Tuple[int, int, int]] = 7,
            max_displacement: Union[float, Tuple[float, float, float]] = 7.5,
            locked_borders: int = 2,
            image_interpolation: str = 'linear',
            p: float = 1,
            seed: Optional[int] = None,
            max_output_channels = 10,
            verbose = False,
            prudent=True
            ):
        def get_torchio_transformer(mask=False):
            if mask:
                interpolation = 'linear'
            else:
                interpolation = image_interpolation
            return RandomElasticDeformation(num_control_points=num_control_points, max_displacement=max_displacement,
                                            locked_borders=locked_borders, image_interpolation=interpolation, p=p,
                                            seed=seed)
        super().__init__(get_transformer=get_torchio_transformer, max_output_channels=max_output_channels, verbose=verbose, prudent=prudent)


class RandomAffineTransform(TorchIOTransformer):
    def __init__(
            self,
            scales: Tuple[float, float] = (0.9, 1.1),
            degrees = 10,
            translation = 0,
            center: str = 'image',
            isotropic: bool = False,
            default_pad_value: Union[str, float] = 'otsu',
            image_interpolation: str = 'linear',
            p: float = 1,
            seed: Optional[int] = None,
            max_output_channels=10,
            verbose = False,
            prudent=True
    ):
        def get_torchio_transformer(mask=False):
            if mask:
                interpolation = 'linear'
            else:
                interpolation = image_interpolation
            return RandomAffine(scales=scales, degrees=degrees, translation=translation, isotropic=isotropic,
                                center=center, default_pad_value=default_pad_value, image_interpolation=interpolation,
                                p=p, seed=seed)
        super().__init__(get_transformer=get_torchio_transformer, max_output_channels=max_output_channels, verbose=verbose, prudent=prudent)


class RandomFlipTransform(TorchIOTransformer):
    def __init__(
            self,
            axes: Union[int, Tuple[int, ...]] = 0,
            flip_probability: float = 0.5,
            p: float = 1,
            seed: Optional[int] = None,
            max_output_channels=10,
            verbose = False,
            prudent=True
    ):
        def get_torchio_transformer(mask=False):
            return RandomFlip(axes=axes, flip_probability=flip_probability, p=p, seed=seed)
        super().__init__(get_transformer=get_torchio_transformer, max_output_channels=max_output_channels, verbose=verbose, prudent=prudent)


class RandomNoiseTransform(TorchIOTransformer):
    def __init__(
            self,
            mean: Union[float, Tuple[float, float]] = 0,
            std: Tuple[float, float] = (0, 0.25),
            p: float = 1,
            seed: Optional[int] = None,
            max_output_channels=10,
            prudent=True
    ):
        def get_torchio_transformer(mask=False):
            if mask:
                # Don't apply noise on mask
                proba = 0
            else:
                proba = p
            return RandomNoise(mean=mean, std=std, p=proba, seed=seed)
        super().__init__(get_transformer=get_torchio_transformer, max_output_channels=max_output_channels, prudent=prudent)