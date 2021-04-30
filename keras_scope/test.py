from tensorflow.keras import backend as K
import tensorflow as tf
from keras_scope.datasets.gsd_outcome_dataset import get_gsd_outcome_dataset
from keras_scope.model import get_model


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

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
        metrics=["acc", "AUC", f1_m, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    model.load_weights(model_path)
    result = model.evaluate(test_dataset)

    print(dict(zip(model.metrics_names, result)))

if __name__ == '__main__':
    test()