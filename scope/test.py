import numpy as np
import tensorflow as tf
from datasets.gsd_outcome_dataset import get_gsd_outcome_dataset
from scope.utils.metrics import f1_m
from scope.models.get_model import get_model

precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()


def test(model_paths, model_type, label_file_path, imaging_dataset_path, outcome, channels, desired_shape,
         single_subject_predictions=False, id_variable='pid', continuous_outcome=False, weight_decay_coefficient=None):
    test_ratio = 1
    batch_size = 200

    _, test_dataset, id_allocation = get_gsd_outcome_dataset(label_file_path, imaging_dataset_path,
                                                             outcome, channels,
                                                             desired_shape, test_ratio, batch_size, id_variable,
                                                             continuous_outcome=continuous_outcome)

    # Frame as regression vs classification
    if continuous_outcome:
        metrics = ['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error']
    else:
        metrics = ["acc", 'AUC', f1_m]

    if type(model_paths) == list and len(model_paths) > 1:
        # if multiple models given use them as an ensemble
        models = np.repeat(get_model(width=desired_shape[0], height=desired_shape[1], depth=desired_shape[2],
                                     channels=len(channels), model_type=model_type, regression=continuous_outcome,
                                     weight_decay_coefficient=weight_decay_coefficient),
                           len(model_paths))
        for model_path, model in zip(model_paths, models):
            model.load_weights(model_path)
        model_input = tf.keras.Input(shape=(desired_shape[0], desired_shape[1], desired_shape[2], len(channels)))
        model_outputs = [model(model_input) for model in models]
        ensemble_output = tf.keras.layers.Average()(model_outputs)
        model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)
    else:
        # otherwise use only a single model
        model = get_model(width=desired_shape[0], height=desired_shape[1], depth=desired_shape[2],
                          n_channels=len(channels), model_type=model_type, regression=continuous_outcome,
                          weight_decay_coefficient=weight_decay_coefficient)
        model.load_weights(model_paths)

    model.compile(
        metrics=metrics,
    )

    result = model.evaluate(test_dataset)
    print(dict(zip(model.metrics_names, result)))

    if single_subject_predictions:
        _, test_ids = id_allocation
        test_subject_predictions = model.predict(test_dataset).ravel()
        test_labels = [a for temp in list(zip(*test_dataset))[1] for a in temp.numpy()]
        subject_prediction_label = np.stack([test_ids, test_subject_predictions, test_labels])
        return dict(zip(model.metrics_names, result)), subject_prediction_label
    return dict(zip(model.metrics_names, result))
