from batchgenerators.transforms import SpatialTransform, \
    GammaTransform, MirrorTransform, Compose, NumpyToTensor
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
import numpy as np


default_3D_augmentation_params = {
    "patch_size": None,

    "do_elastic": True,
    "elastic_deform_alpha": (0., 100.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.5,

    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis": 1,
    "p_scale": 0.5,

    "do_rotation": True,
    "rotation_x": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_y": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_z": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.5,

    "random_crop": True,
    "random_crop_dist_to_border": (10, 10, 10),

    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,

    "do_mirror": True,
    "mirror_axes": (0, 1, 2),

    "dummy_2D": False,
    "mask_was_used_for_normalization": False,
    "border_mode_data": "constant",

    "do_additive_brightness": True,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,
}


def get_DKFZ_augmentation_sequence(params=default_3D_augmentation_params,
                              border_val_seg=-1,
                              order_seg=1, order_data=3):
    tr_transforms = []

    tr_transforms.append(SpatialTransform(
        params.get("patch_size"), patch_center_dist_from_border=params.get("random_crop_dist_to_border"),
        do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
        sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
        do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

    if params.get("do_additive_brightness"):
        tr_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
                                                 params.get("additive_brightness_sigma"),
                                                 True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                                 p_per_channel=params.get("additive_brightness_p_per_channel")))

    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=None))
    tr_transforms.append(
        GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                       p_per_sample=0.1))  # inverted gamma

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    tr_transforms = Compose(tr_transforms)

    val_transforms = []
    val_transforms = Compose(val_transforms)

    return tr_transforms, val_transforms
