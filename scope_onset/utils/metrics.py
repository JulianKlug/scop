import tensorflow as tf


class RegressionAUC(tf.keras.metrics.Metric):
    def __init__(self, name='AUC', **kwargs):
        super(RegressionAUC, self).__init__(name=name, **kwargs)
        self.internal_model = tf.keras.metrics.AUC()

    def update_state(self, y_true, y_pred):
        # binarize using 4.5h threshold
        y_true = y_true / 60 > 4.5
        y_pred = y_pred / 60 > 4.5

        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        self.internal_model.update_state(y_true, y_pred)

    def result(self):
        return self.internal_model.result()

    def reset_state(self):
        self.internal_model.reset_state()
