import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras import backend as K
import tensorflow_probability as tfp

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


class RegressionAUC(tf.keras.metrics.Metric):
    def __init__(self, name='auc', **kwargs):
        super(RegressionAUC, self).__init__(name=name, **kwargs)
        self.internal_model = tf.keras.metrics.AUC()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # binarize using 4.5h threshold
        y_true = y_true / 60 > 4.5
        y_pred = y_pred / 60 > 4.5

        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        self.internal_model.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.internal_model.result()

    def reset_state(self):
        self.internal_model.reset_state()


class RunningAUC(tf.keras.metrics.Metric):

  def __init__(self, name='running_auc', history_length=10, **kwargs):
    super(RunningAUC, self).__init__(name=name, **kwargs)
    self.internal_model = tf.keras.metrics.AUC()
    self.history_length = history_length
    self.running_history = []

  @tf.function
  def update_state(self, y_true, y_pred, sample_weight=None):
      self.internal_model.update_state(y_true, y_pred, sample_weight)

  @tf.function
  def result(self):
    current_epoch_with_history = [self.internal_model.result().numpy()] + self.running_history
    if len(self.running_history) == 0:
        return self.internal_model.result()
    return tfp.stats.percentile(current_epoch_with_history, 50.0, interpolation='midpoint')

  @tf.function
  def reset_state(self):
      self.running_history.append(self.internal_model.result().numpy())
      if len(self.running_history) > self.history_length:
          self.running_history.pop()
      self.internal_model.reset_state()
