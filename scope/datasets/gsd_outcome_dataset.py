import numpy as np
import tensorflow as tf
from preprocessing import min_max_normalize, resize_volume
import pandas as pd
from scope.datasets.base_dataset import base_dataset
from scope.utils.augmentations import RandAugment3D


augment = RandAugment3D(n=2)

def image_preprocessing(min, max, desired_shape):
    def process(volume):
        volume = min_max_normalize(volume, min, max)
        # Resize width, height and depth
        volume = resize_volume(volume, desired_shape[0], desired_shape[1], desired_shape[2])
        return volume

    return process

def train_augmentation(volume, label):
    """Process training data by rotating and adding a channel."""
    volume = tf.numpy_function(augment, [volume], tf.float32)
    volume = tf.convert_to_tensor(volume)
    return volume, label

def no_augmentation(volume, label):
    """Do not apply any augmentation"""
    # volume = tf.expand_dims(volume, axis=3)
    return volume, label

def validation_augmentation(volume, label):
    """Process validation data by only adding a channel."""
    # volume = tf.expand_dims(volume, axis=3)
    return volume, label


def get_gsd_outcome_dataset(label_file_path, imaging_dataset_path, outcome, channels, desired_shape, test_ratio,
                            batch_size, id_variable, continuous_outcome=False, use_augmentation=True):
    params = np.load(imaging_dataset_path, allow_pickle=True)['params']
    try:
        print('Using channels:', [params.item()['ct_sequences'][channel] for channel in channels])
    except:
        print('Geneva Stroke Dataset (perfusion CT maps) parameters: ', params)

    ids = np.load(imaging_dataset_path, allow_pickle=True)['ids']

    outcomes_df = pd.read_excel(label_file_path)
    labels = np.array([outcomes_df.loc[outcomes_df[id_variable] == subj_id, outcome].iloc[0] for
                       subj_id in ids])

    raw_images = np.load(imaging_dataset_path, allow_pickle=True)['ct_inputs'][..., channels]

    if raw_images.ndim < 5:
        raw_images = np.expand_dims(raw_images, axis=-1)

    # Apply masks
    raw_masks = np.load(imaging_dataset_path, allow_pickle=True)['brain_masks']
    raw_masks = np.expand_dims(raw_masks, axis=-1)
    images = raw_images * raw_masks

    if use_augmentation:
        applied_train_augmentation = train_augmentation
    else:
        applied_train_augmentation = no_augmentation

    train_dataset, validation_dataset, id_allocation = base_dataset(images, labels, test_ratio, batch_size,
                                                                     image_preprocessing(min=0, max=400,
                                                                                         desired_shape=desired_shape),
                                                                     applied_train_augmentation, validation_augmentation,
                                                                     ids=ids, continuous_outcome=continuous_outcome)

    return train_dataset, validation_dataset, id_allocation
