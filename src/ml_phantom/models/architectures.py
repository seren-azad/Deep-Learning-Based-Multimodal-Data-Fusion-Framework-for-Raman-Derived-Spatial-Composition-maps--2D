from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def build_model1(input_shape=(150, 150, 1), num_phases: int = 6, grid_cells: int = 6) -> tf.keras.Model:
    """Basic encoder model from the notebook.

    The network extracts image features through convolutional blocks, then uses
    global average pooling and a dense head to predict a ``grid_cells × grid_cells × num_phases``
    fraction tensor.
    """

    inputs = tf.keras.Input(shape=input_shape)

    x = _conv_block(inputs, 64, dropout=0.1)
    x = layers.MaxPooling2D((2, 2))(x)
    x = _conv_block(x, 128, dropout=0.1)
    x = layers.MaxPooling2D((2, 2))(x)
    x = _conv_block(x, 256, dropout=0.2)
    x = layers.MaxPooling2D((2, 2))(x)
    x = _conv_block(x, 512, dropout=0.2)
    x = layers.MaxPooling2D((2, 2))(x)

    x = _conv_block(x, 1024, dropout=0.3)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(grid_cells * grid_cells * num_phases, activation="linear")(x)
    outputs = layers.Reshape((grid_cells, grid_cells, num_phases))(x)

    return models.Model(inputs=inputs, outputs=outputs, name="model1_basic_encoder")


def build_model2(input_shape=(150, 150, 1), num_phases: int = 6, grid_cells: int = 6) -> tf.keras.Model:
    """Residual encoder with a multiplicative grid-attention head."""

    inputs = tf.keras.Input(shape=input_shape)
    x = residual_block(inputs, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)
    x = residual_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)
    x = residual_block(x, 256)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 512)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 1024)
    x = layers.Dropout(0.3)(x)

    outputs = _attention_grid_head(x, grid_cells=grid_cells, num_phases=num_phases)
    return models.Model(inputs=inputs, outputs=outputs, name="model2_residual_attention")


def build_model3(input_shape=(150, 150, 1), num_phases: int = 6, grid_cells: int = 6) -> tf.keras.Model:
    """Enhanced residual encoder with squeeze-and-excitation attention."""

    inputs = tf.keras.Input(shape=input_shape)
    x = se_residual_block(inputs, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.1)(x)
    x = se_residual_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.1)(x)
    x = se_residual_block(x, 256)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.2)(x)
    x = se_residual_block(x, 512)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.2)(x)
    x = se_residual_block(x, 1024)
    x = layers.SpatialDropout2D(0.3)(x)

    outputs = _attention_grid_head(x, grid_cells=grid_cells, num_phases=num_phases)
    return models.Model(inputs=inputs, outputs=outputs, name="model3_se_residual_attention")


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3, use_decay: bool = False) -> tf.keras.Model:
    """Compile a model for fraction regression."""

    if use_decay:
        learning_rate = ExponentialDecay(learning_rate, decay_steps=142, decay_rate=0.96, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mean_absolute_error"])
    return model


def get_model_builder(name: str):
    name = name.lower().strip()
    builders = {
        "model1": build_model1,
        "model2": build_model2,
        "model3": build_model3,
    }
    if name not in builders:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(builders)}")
    return builders[name]


def _conv_block(x, filters: int, dropout: float = 0.0):
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x


def residual_block(x, filters: int, kernel_size=(3, 3), padding="same", strides=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = layers.BatchNormalization()(x)

    shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([x, shortcut])
    return layers.Activation("relu")(x)


def se_residual_block(x, filters: int, kernel_size=(3, 3), padding="same", strides=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = layers.BatchNormalization()(x)

    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(filters // 16, 1), activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, filters))(se)
    x = layers.multiply([x, se])

    shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([x, shortcut])
    return layers.Activation("relu")(x)


def _attention_grid_head(x, grid_cells: int, num_phases: int):
    pooled = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(512, activation="relu")(pooled)
    attention = layers.Dense(grid_cells * grid_cells * num_phases, activation="sigmoid")(attention)
    attention = layers.Reshape((grid_cells, grid_cells, num_phases))(attention)

    dense = layers.Dense(grid_cells * grid_cells * num_phases, activation="linear")(pooled)
    dense = layers.Reshape((grid_cells, grid_cells, num_phases))(dense)
    return layers.Multiply()([dense, attention])
