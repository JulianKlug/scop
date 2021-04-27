import numpy as np
from keras_scope.augmentation import rotate
from keras_scope.preprocessing import min_max_normalize, resize_volume
import tensorflow as tf
import pandas as pd
from keras_scope.datasets.base_dataset import base_dataset


def image_preprocessing(min, max, desired_shape):
    def process(volume):
        volume = min_max_normalize(volume, min, max)
        # Resize width, height and depth
        volume = resize_volume(volume, desired_shape[0], desired_shape[1], desired_shape[2], with_channels=False)
        return volume
    return process


def train_augmentation(volume, label):
    """Process training data by rotating and adding a channel."""
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_augmentation(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def get_LeftRightDataset(label_file_path, data_file_path, desired_shape, split_ratio, batch_size):

    outcomes_df = pd.read_csv(label_file_path, index_col=0)
    labels = np.array(outcomes_df).squeeze()
    images = np.load(data_file_path)

    train_dataset, validation_dataset = base_dataset(images, labels, split_ratio, batch_size,
                                                     image_preprocessing(min=0, max=1, desired_shape=desired_shape),
                                                     train_augmentation, validation_augmentation)

    return train_dataset, validation_dataset
