# set fixed seed
seed_value= 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)

from modun.file_io import dict2json
from scope_onset.utils.utils import ensure_dir
from scope_onset.models.resnet3d import Resnet3DBuilder
from scope_onset.parse_config import parse_config
from datetime import datetime
from tensorflow import keras
# from datasets.leftright_dataset import get_LeftRightDataset
from datasets.gsd_outcome_dataset import get_gsd_outcome_dataset
from metrics import f1_m
from scope_onset.models.basic_model import get_regression_model, get_classification_model


def train(label_file_path, imaging_dataset_path, main_log_dir, outcome, channels, model_input_shape,
          initial_learning_rate, id_variable, continuous_outcome=False, epochs=200, early_stopping_patience=100, split_ratio=0.3, batch_size=2,
          target_metric='max auc', use_augmentation=True, force_cpu=False, weight_decay_coefficient=1e-4,
          logdir=None):

    if force_cpu:
        print("Disabling GPUs.")
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

    # train_dataset, validation_dataset = get_LeftRightDataset(label_file_path, imaging_dataset_path, model_input_shape, split_ratio, batch_size)
    train_dataset, validation_dataset, id_allocation = get_gsd_outcome_dataset(label_file_path, imaging_dataset_path,
                                                                outcome, channels,
                                                                model_input_shape, split_ratio, batch_size, id_variable,
                                                                               continuous_outcome=continuous_outcome,
                                                                               use_augmentation=use_augmentation)

    # Frame as regression vs classification
    if continuous_outcome:
        get_model = get_regression_model
        loss = "mse"
        metrics = ['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error']
    else:
        loss = "binary_crossentropy"
        metrics = ["acc", 'AUC', f1_m]
        get_model = get_classification_model

    # Build model.
    # model = get_model(width=model_input_shape[0], height=model_input_shape[1], depth=model_input_shape[2],
    #                   channels=len(channels), weight_decay=weight_decay_coefficient)
    model = Resnet3DBuilder().build_resnet_50((model_input_shape[0], model_input_shape[1], model_input_shape[2], len(channels)),
                                              1, regression=continuous_outcome, reg_factor=weight_decay_coefficient)

    # import models.efficientnet_3D.tfkeras as efn
    # model = efn.EfficientNetB0(input_shape=
    #                            (model_input_shape[0], model_input_shape[1], model_input_shape[2], len(channels)),
    #                             weights=None, classes=1, include_top=True, regression=continuous_outcome)
    model.summary()

    # Learning rate decay
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=config.lr_decay_steps, decay_rate=0.96,
        staircase=True
    )
    # Compile model.
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        # optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        # optimizer='adam',
        metrics=metrics,
    )

    target_metric = target_metric.split()

    # Define callbacks.
    if logdir is None:
        logdir = os.path.join(main_log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    model_path = os.path.join(logdir, '3d_model.h5')
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        model_path, save_best_only=True,
        monitor='val_' + target_metric[1], mode=target_metric[0]
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_"+target_metric[1],
        min_delta=1,
        patience=early_stopping_patience,
        mode=target_metric[0])
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=5,
    )

    # Train the model, doing validation at the end of each epoch
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_callback],
    )

    return model, model_path


if __name__ == '__main__':
    config = parse_config()
    comment = config.comment
    logdir = os.path.join(config.output_dir,
                          datetime.now().strftime("%Y%m%d_%H%M%S")
                          + comment)
    ensure_dir(logdir)
    dict2json(os.path.join(logdir, 'config.json'), vars(config))


    train(config.label_file_path, config.imaging_dataset_path, config.output_dir,
          config.outcome, config.channels, config.model_input_shape,
          config.initial_learning_rate, config.id_variable,
          continuous_outcome=config.continuous_outcome,
          epochs=config.epochs,
          early_stopping_patience=config.early_stopping_patience,
          split_ratio=config.validation_size,
          batch_size=config.batch_size,
          target_metric=config.target_metric,
          use_augmentation=config.use_augmentation,
          weight_decay_coefficient=config.weight_decay_coefficient,
          logdir=logdir)
