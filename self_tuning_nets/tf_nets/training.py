from typing import List

import numpy as np
import tensorflow as tf


def set_weights(
    model: tf.keras.Model,
    seed: int
) -> List[np.ndarray]:
    """We can fix the weights from given seed"""
    parameters_size = 0
    for array in model.get_weights():
        parameters_size += array.size

    # set weights from uniform distribution in [-1, 1)
    rng = np.random.RandomState(seed)
    seed_weights = 2.0 * rng.random(parameters_size).astype(np.float32) - 1.0

    start, end = 0, 0
    new_weights = []
    for array in model.get_weights():
        end = start + array.size
        new_array = seed_weights[start:end].reshape(array.shape)
        shape = new_array.shape
        # pytorch inverts the matmul operands so we need the transpose
        # to have the same representation of seed generated weights
        if len(shape) > 1 and shape[0] > 1 and shape[1] > 1:
            new_array = np.transpose(new_array)
        new_weights.append(new_array)
        start = end

    model.set_weights(new_weights)
    return model.get_weights()


def train_step(model: tf.keras.Model, X: np.ndarray):
    loss = model.train_on_batch(x=X, y=(X**2))
    return model, loss


def eval_step(model: tf.keras.Model, X: np.ndarray):
    pred = model.predict_on_batch(x=X)
    loss = model.test_on_batch(x=X, y=(X**2))
    return loss, pred
