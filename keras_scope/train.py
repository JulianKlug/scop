from datetime import datetime
import os
from tensorflow import keras
# from datasets.leftright_dataset import get_LeftRightDataset
from datasets.gsd_outcome_dataset import get_gsd_outcome_dataset
from model import get_model


def main():
    # label_file_path = '/Users/jk1/stroke_research/SimpleVoxel-3D/leftright/labels.csv'
    # imaging_dataset_path = '/Users/jk1/stroke_research/SimpleVoxel-3D/leftright/data.npy'
    # desired_shape = (46, 46, 46)

    label_file_path = '/home/klug/working_data/clinical/clinical_outcome/joined_anon_outcomes_2015_2016_2017_2018_df.xlsx'
    imaging_dataset_path = '/home/klug/working_data/perfusion_maps/no_GT/train_noGT_pmaps_15-19_dataset_with_combined_mRS_90_days.npz'

    main_log_dir = '/home/klug/output/keras_scope'
    channels = [0, 1, 2, 3]
    outcome = "combined_mRS_0-2_90_days"
    desired_shape = (46, 46, 46)
    # desired_shape = (79, 95, 70)


    split_ratio = 0.3
    batch_size = 2
    epochs = 200
    initial_learning_rate = 0.0001

    # train_dataset, validation_dataset = get_LeftRightDataset(label_file_path, imaging_dataset_path, desired_shape, split_ratio, batch_size)
    train_dataset, validation_dataset = get_gsd_outcome_dataset(label_file_path, imaging_dataset_path,
                                                                outcome, channels,
                                                                desired_shape, split_ratio, batch_size)


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
        metrics=["acc"],
    )

    # Define callbacks.
    logdir = os.path.join(main_log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        os.path.join(logdir, "3d_image_classification.h5"), save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
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
