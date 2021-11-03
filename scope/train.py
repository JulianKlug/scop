# set fixed seed
seed_value= 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)

import GPUtil
try:
    # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Get the first available GPU
    DEVICE_ID_LIST = GPUtil.getFirstAvailable()
    DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list
    # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    print('Using GPU', DEVICE_ID)
except:
    print('No GPU found.')
import tensorflow as tf
tf.random.set_seed(seed_value)

from modun.file_io import dict2json
from scope.utils.utils import ensure_dir
from scope.parse_config import parse_config
from datetime import datetime
from tensorflow import keras
# from datasets.leftright_dataset import get_LeftRightDataset
from datasets.gsd_outcome_dataset import get_gsd_outcome_dataset
from scope.utils.metrics import f1_m
from scope.utils.metrics import RegressionAUC
from scope.models.get_model import get_model


def train(label_file_path, imaging_dataset_path, main_log_dir, outcome, channels, model_type, model_input_shape,
          initial_learning_rate, id_variable, continuous_outcome=False, epochs=200, early_stopping_patience=100, split_ratio=0.3, batch_size=2,
          target_metric='max auc', use_augmentation=True, augmentation_magnitude=10, force_cpu=False, weight_decay_coefficient=1e-4, lr_decay_steps=10000000,
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
                                                                               use_augmentation=use_augmentation,
                                                                               augmentation_magnitude=augmentation_magnitude)

    # Frame as regression vs classification
    if continuous_outcome:
        loss = "mse"
        metrics = ['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error', RegressionAUC()]
    else:
        loss = "binary_crossentropy"
        metrics = ["acc", 'AUC', f1_m]

    # Build model.
    model = get_model(width=model_input_shape[0], height=model_input_shape[1], depth=model_input_shape[2],
                      n_channels=len(channels), model_type=model_type, weight_decay_coefficient=weight_decay_coefficient,
                      regression=continuous_outcome)
    model.summary()

    # Learning rate decay
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=lr_decay_steps, decay_rate=0.96,
        staircase=True
    )
    # Compile model.
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=metrics,
    )

    target_metric = target_metric.split()

    # Define callbacks.
    if logdir is None:
        logdir = os.path.join(main_log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        os.path.join(logdir, '3d_model_{epoch}.h5'), save_best_only=True,
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
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_callback],
    )

    best_val_score_index = np.argmax(history.history["val_" + target_metric[1]])
    plateau_radius = 5
    try:
        best_val_score_plateau = np.mean(history.history["val_" + target_metric[1]]
                                         [best_val_score_index-plateau_radius:best_val_score_index+plateau_radius])
    except:
        best_val_score_plateau = history.history["val_" + target_metric[1]][best_val_score_index]

    # only retain best model
    saved_model_list = [file for file in os.listdir(logdir) if file.endswith('.h5')]
    best_model_index = np.argmax([int(file.split('.')[0].split('_')[-1]) for file in saved_model_list])
    best_model_path = os.path.join(logdir, saved_model_list[best_model_index])
    best_saved_epoch = int(best_model_path.split('.')[-2].split('_')[-1])
    # delete all other saved models
    saved_model_list.remove(saved_model_list[best_model_index])
    for model_file in saved_model_list:
        os.remove(os.path.join(logdir, model_file))

    return model, best_model_path, best_saved_epoch, best_val_score_plateau


if __name__ == '__main__':
    config = parse_config()
    comment = config.comment
    logdir = os.path.join(config.output_dir,
                          datetime.now().strftime("%Y%m%d_%H%M%S")
                          + comment)
    ensure_dir(logdir)
    dict2json(os.path.join(logdir, 'config.json'), vars(config))


    train(config.label_file_path, config.imaging_dataset_path, config.output_dir,
          config.outcome, config.channels, config.model_type, config.model_input_shape,
          config.initial_learning_rate, config.id_variable,
          continuous_outcome=config.continuous_outcome,
          epochs=config.epochs,
          early_stopping_patience=config.early_stopping_patience,
          split_ratio=config.validation_size,
          batch_size=config.batch_size,
          target_metric=config.target_metric,
          use_augmentation=config.use_augmentation,
          augmentation_magnitude=config.augmentation_magnitude,
          weight_decay_coefficient=config.weight_decay_coefficient,
          lr_decay_steps=config.lr_decay_steps,
          logdir=logdir)
