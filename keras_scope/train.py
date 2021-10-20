from datetime import datetime
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
# from datasets.leftright_dataset import get_LeftRightDataset
from datasets.gsd_outcome_dataset import get_gsd_outcome_dataset
from metrics import f1_m
from model import get_model


def train(label_file_path, imaging_dataset_path, main_log_dir, outcome, channels, desired_shape,
          initial_learning_rate, epochs=200, split_ratio=0.3, batch_size=2,
          monitoring_metric='auc', id_variable='pid', early_stopping_patience=100, force_cpu=False,
          use_augmentation=True):

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

    # train_dataset, validation_dataset = get_LeftRightDataset(label_file_path, imaging_dataset_path, desired_shape, split_ratio, batch_size)
    train_dataset, validation_dataset, id_allocation = get_gsd_outcome_dataset(label_file_path, imaging_dataset_path,
                                                                outcome, channels,
                                                                desired_shape, split_ratio, batch_size,
                                                                               use_augmentation=use_augmentation,
                                                                               id_variable=id_variable)


    # Build model.
    model = get_model(width=desired_shape[0], height=desired_shape[1], depth=desired_shape[2], channels=len(channels))
    model.summary()

    # Compile model.
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc", 'AUC', f1_m],
    )

    # Define callbacks.
    logdir = os.path.join(main_log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_path = os.path.join(logdir, "3d_image_classification.h5")
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        model_path, save_best_only=True,
        monitor='val_' + monitoring_metric, mode='max'
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_"+monitoring_metric,
                                                      patience=early_stopping_patience, mode='max')
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

    best_val_score_index = np.argmax(history.history["val_" + monitoring_metric])
    plateau_radius = 5
    try:
        best_val_score_plateau = np.mean(history.history["val_" + monitoring_metric]
                                         [best_val_score_index-plateau_radius:best_val_score_index+plateau_radius])
    except:
        best_val_score_plateau = history.history["val_" + monitoring_metric][best_val_score_index]

    return model, model_path, best_val_score_plateau


if __name__ == '__main__':
    # label_file_path = '/Users/jk1/stroke_research/SimpleVoxel-3D/leftright/labels.csv'
    # imaging_dataset_path = '/Users/jk1/stroke_research/SimpleVoxel-3D/leftright/data.npy'
    # desired_shape = (46, 46, 46)

    # MIP lab config
    # label_file_path = '/home/klug/working_data/clinical/clinical_outcome/joined_anon_outcomes_2015_2016_2017_2018_df.xlsx'
    # imaging_dataset_path = '/home/klug/working_data/perfusion_maps/no_GT/train_noGT_pmaps_15-19_dataset_with_combined_mRS_90_days.npz'
    # main_log_dir = '/home/klug/output/keras_scope'

    label_file_path = '/mnt/data/hendrik/jk/scope_data/joined_anon_outcomes_2015_2016_2017_2018_df.xlsx'
    imaging_dataset_path = '/mnt/data/hendrik/jk/scope_data/train_noGT_pmaps_15-19_dataset_with_combined_mRS_90_days.npz'
    main_log_dir = '/home/hendrik/jk/output/keras_scope'
    outcome = "combined_mRS_0-2_90_days"

    channels = [0, 1, 2, 3]
    desired_shape = (46, 46, 46)
    # desired_shape = (79, 95, 70)

    epochs = 400
    initial_learning_rate = 0.0001

    train(label_file_path, imaging_dataset_path, main_log_dir, outcome, channels, desired_shape,
          initial_learning_rate, epochs)
