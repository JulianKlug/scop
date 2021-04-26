from datetime import datetime
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt

from augmentation import rotate
from model import get_model
from preprocessing import min_max_normalize, resize_volume


def image_preprocessing(volume, min, max, desired_shape):
    volume = min_max_normalize(volume, min, max)
    # Resize width, height and depth
    volume = resize_volume(volume, desired_shape[0], desired_shape[1], desired_shape[2])
    return volume


def train_augmentation(volume, label):
    """Process training data by rotating and adding a channel."""
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_augmentation(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def main():
    label_file_path = '/home/klug/working_data/leftright_dataset/labels.csv'
    data_file_path = '/home/klug/working_data/leftright_dataset/data.npy'
    main_log_dir = '/home/klug/output/leftright'

    # label_file_path = '/Users/jk1/stroke_research/SimpleVoxel-3D/leftright/labels.csv'
    # data_file_path = '/Users/jk1/stroke_research/SimpleVoxel-3D/leftright/data.npy'
    desired_shape = (46, 46, 46)
    split_ratio = 0.3
    batch_size = 2
    epochs = 20
    initial_learning_rate = 0.0001

    outcomes_df = pd.read_csv(label_file_path, index_col=0)
    labels = np.array(outcomes_df).squeeze()

    images = np.load(data_file_path)

    images = np.array([image_preprocessing(image,  min=0, max=1, desired_shape=desired_shape) for image in images])

    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=split_ratio, random_state=42,
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

    # Build model.
    model = get_model(width=desired_shape[0], height=desired_shape[1], depth=desired_shape[2])
    model.summary()

    # Compile model.
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "3d_image_classification.h5", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
    logdir = os.path.join(main_log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    # Train the model, doing validation at the end of each epoch
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_callback],
    )


if __name__ == '__main__':
    main()
