from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def get_classification_model(width=128, height=128, depth=64, channels=1, weight_decay=0):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, channels))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


def get_regression_model(width=128, height=128, depth=64, channels=1, weight_decay=0):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, channels))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="linear")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


