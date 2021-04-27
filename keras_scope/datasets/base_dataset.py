from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

def base_dataset(images, labels, test_ratio, batch_size,
                 image_preprocessing, train_augmentation, validation_augmentation):
    images = np.array([image_preprocessing(image) for image in images])

    if test_ratio == 1:
        # all data goes into testing/validation
        data_loader = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = (
            data_loader.shuffle(len(images))
                .map(validation_augmentation)
                .batch(batch_size)
                .prefetch(2)
        )
        return None, dataset

    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=test_ratio, random_state=42,
                                                      shuffle=True, stratify=labels)

    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # Augment the on the fly during training.
    train_dataset = (
        train_loader.shuffle(len(x_train))
            .map(train_augmentation)
            .batch(batch_size)
            .prefetch(2)
    )
    # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(len(x_val))
            .map(validation_augmentation)
            .batch(batch_size)
            .prefetch(2)
    )

    return train_dataset, validation_dataset