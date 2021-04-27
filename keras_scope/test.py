import numpy as np

from keras_scope.datasets.gsd_outcome_dataset import get_gsd_outcome_dataset
from keras_scope.model import get_model


def test():
    model_path = '/Users/jk1/Downloads/3d_image_classification.h5'
    label_file_path = '/Users/jk1/temp/scope_test/joined_anon_outcomes_2015_2016_2017_2018_df.xlsx'
    imaging_dataset_path = '/Users/jk1/stroke_datasets/dataset_files/perfusion_data_sets/noGT_datasets/test_noGT_pmaps_15-19_dataset_with_combined_mRS_90_days.npz'
    desired_shape = (46, 46, 46)
    channels = [0, 1, 2, 3]
    test_ratio = 1
    outcome = "combined_mRS_0-2_90_days"
    batch_size = 200

    _, test_dataset = get_gsd_outcome_dataset(label_file_path, imaging_dataset_path,
                                                                outcome, channels,
                                                                desired_shape, test_ratio, batch_size)


    model = get_model(width=desired_shape[0], height=desired_shape[1], depth=desired_shape[2], channels=len(channels))

    model.compile(
        metrics=["acc", "AUC"],
    )

    model.load_weights(model_path)
    result = model.evaluate(test_dataset)

    print(dict(zip(model.metrics_names, result)))

if __name__ == '__main__':
    test()