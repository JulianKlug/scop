from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

from scope.utils.data_splitting import continuously_stratified_train_test_split


def base_dataset(images, labels, test_ratio, batch_size,
                 image_preprocessing, train_augmentation, validation_augmentation,
                 ids=None, continuous_outcome=False):
    images = np.array([image_preprocessing(image) for image in images])

    if test_ratio == 1:
        # all data goes into testing/validation
        data_loader = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = (
            data_loader
                .map(validation_augmentation)
                .batch(batch_size)
                .prefetch(2)
        )
        return None, dataset, (None, ids)

    if continuous_outcome:
        split_function = continuously_stratified_train_test_split
    else:
        split_function = train_test_split

    if ids is not None:
        ids_train, ids_val, x_train, x_val, y_train, y_val = split_function(
            ids, images, labels, test_size=test_ratio, random_state=42,
            shuffle=True, stratify=labels)
    else:
        x_train, x_val, y_train, y_val = split_function(images, labels, test_size=test_ratio, random_state=42,
                                                          shuffle=True, stratify=labels)
        ids_train, ids_val = None, None

    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # Augment the on the fly during training.
    train_dataset = (
        train_loader
            .map(train_augmentation)
            .batch(batch_size)
            .prefetch(2)
    )
    # Only rescale.
    validation_dataset = (
        validation_loader
            .map(validation_augmentation)
            .batch(batch_size)
            .prefetch(2)
    )

    return train_dataset, validation_dataset, (ids_train, ids_val)
