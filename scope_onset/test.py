import numpy as np
import tensorflow as tf
from datasets.gsd_outcome_dataset import get_gsd_outcome_dataset
from metrics import f1_m
from scope_onset.models.basic_model import get_regression_model, get_classification_model

precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()


def test(model_path, label_file_path, imaging_dataset_path, outcome, channels, desired_shape, id_variable,
         single_subject_predictions = False, continuous_outcome=False):
    test_ratio = 1
    batch_size = 200

    _, test_dataset, id_allocation = get_gsd_outcome_dataset(label_file_path, imaging_dataset_path,
                                              outcome, channels,
                                              desired_shape, test_ratio, batch_size, id_variable,
                                                             continuous_outcome=continuous_outcome)

    # Frame as regression vs classification
    if continuous_outcome:
        get_model = get_regression_model
        metrics = ['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error']
    else:
        metrics = ["acc", 'AUC', f1_m]
        get_model = get_classification_model

    model = get_model(width=desired_shape[0], height=desired_shape[1], depth=desired_shape[2], channels=len(channels))

    model.compile(
        metrics=metrics,
    )

    model.load_weights(model_path)
    result = model.evaluate(test_dataset)
    print(dict(zip(model.metrics_names, result)))

    if single_subject_predictions:
        _, test_ids = id_allocation
        test_subject_predictions = model.predict(test_dataset).ravel()
        test_labels = [a for temp in list(zip(*test_dataset))[1] for a in temp.numpy()]
        subject_prediction_label = np.stack([test_ids, test_subject_predictions, test_labels])
        return dict(zip(model.metrics_names, result)), subject_prediction_label
    return dict(zip(model.metrics_names, result))



