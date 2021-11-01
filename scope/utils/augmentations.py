import random

import numpy as np
import tensorflow as tf
from skimage.exposure import equalize_hist
from monai.transforms import RandGaussianNoise, RandStdShiftIntensity, RandScaleIntensity, RandAdjustContrast, \
    RandGaussianSharpen, RandHistogramShift, RandAffine, Rand3DElastic


def solarize(image: tf.Tensor, threshold: int = 0.5) -> tf.Tensor:
    """Solarize the input image(s)."""
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract max from the pixel.
    return tf.where(image < threshold * tf.reduce_max(image), image, tf.reduce_max(image) - image).numpy()


def equalize(volume):
    return equalize_hist(volume)


def elastic(volume):
    transformer = Rand3DElastic(sigma_range=(5, 7), magnitude_range=(100, 150), prob=1, padding_mode='zeros')
    return transformer(volume)


def rotate_3d(volume):
    # full level in radians (approx 30°)
    full_level = 0.5
    transformer = RandAffine(prob=1, rotate_range=((full_level, -full_level), (full_level, -full_level),
                                                   (full_level, -full_level)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)

def rotate_x(volume):
    # rotation along x axis
    # full level in radians (approx 30°)
    full_level = 0.5
    transformer = RandAffine(prob=1, rotate_range=((full_level, -full_level), (0), (0)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)

def rotate_y(volume):
    # rotation along y axis
    # full level in radians (approx 30°)
    full_level = 0.5
    transformer = RandAffine(prob=1, rotate_range=((0), (full_level, -full_level), (0)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)

def rotate_z(volume):
    # rotation along z axis
    # full level in radians (approx 30°)
    full_level = 0.5
    transformer = RandAffine(prob=1, rotate_range=((0), (0), (full_level, -full_level)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)

def shear_3d(volume):
    full_level = 0.3
    transformer = RandAffine(prob=1, shear_range=((full_level, -full_level), (full_level, -full_level), (full_level, -full_level)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)

def shear_x(volume):
    full_level = 0.3
    transformer = RandAffine(prob=1, shear_range=((full_level, -full_level), (0), (0)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)

def shear_y(volume):
    full_level = 0.3
    transformer = RandAffine(prob=1, shear_range=((0), (full_level, -full_level), (0)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)

def shear_z(volume):
    full_level = 0.3
    transformer = RandAffine(prob=1, shear_range=((0), (0), (full_level, -full_level)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def translate(volume):
    transformer = RandAffine(prob=1, translate_range=((10, -10), (10, -10), (10, -10)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def scale(volume):
    full_level = 0.3
    transformer = RandAffine(prob=1, scale_range=full_level, padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def noise(volume):
    transformer = RandGaussianNoise()
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def shiftIntensity(volume):
    transformer = RandStdShiftIntensity(prob=1, factors=(5, 10))
    return transformer(volume)


def scaleIntensity(volume):
    transformer = RandScaleIntensity(prob=1, factors=(5, 10))
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def adjustContrast(volume):
    transformer = RandAdjustContrast(prob=1)
    return transformer(volume)


def sharpen(volume):
    transformer = RandGaussianSharpen(prob=1)
    return transformer(volume)


def histogramShift(volume):
    transformer = RandHistogramShift(num_control_points=10, prob=1)
    return transformer(volume)


def identity(volume):
    return volume


def to_channels_first(tensor):
    return tf.experimental.numpy.moveaxis(tensor, 3, 0)


def to_channels_last(tensor):
    return tf.experimental.numpy.moveaxis(tensor, 0, 3)


def safe_numpy_conversion(object):
    if type(object) is np.ndarray:
        return object
    else:
        return object.numpy()


def augment_list():
    l = [
        identity,
        # elastic,
        rotate_x,
        rotate_y,
        rotate_z,
        translate,
        scale,
        shear_x,
        shear_y,
        shear_z,
        shiftIntensity,
        equalize,
        solarize,
        histogramShift,
        sharpen,
        adjustContrast,
    ]

    return l


'''
Transforms in original paper: [
    ’Identity’, ’AutoContrast’, ’Equalize’,
    ’Rotate’, ’Solarize’, ’Color’, ’Posterize’,
    ’Contrast’, ’Brightness’, ’Sharpness’,
    ’ShearX’, ’ShearY’, ’TranslateX’, ’TranslateY’]
'''


class RandAugment3D:
    def __init__(self, n):
        self.n = n
        # TODO Implement magnitude
        # self.magnitude = m
        self.augment_list = augment_list()

    def __call__(self, volume):
        volume = to_channels_first(volume).numpy()
        ops = random.sample(self.augment_list, k=self.n)
        for op in ops:
            print(op.__name__)
            volume = op(volume)
        volume = to_channels_last(volume)
        volume = tf.cast(volume, tf.float32)

        return volume
