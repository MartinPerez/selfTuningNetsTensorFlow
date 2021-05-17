import numpy as np
import tensorflow as tf
import torch
from numpy import testing

from self_tuning_nets.tf_nets.deep_relu_net import deep_relu_net
from self_tuning_nets.tf_nets.training import eval_step as keras_eval
from self_tuning_nets.tf_nets.training import set_weights as set_keras_weights
from self_tuning_nets.torch_nets.deep_relu_net import DeepReluNet
from self_tuning_nets.torch_nets.training import eval_step as torch_eval
from self_tuning_nets.torch_nets.training import \
    set_weights as set_torch_weights

# keras style imports
clear_session = tf.keras.backend.clear_session


def theoretical_deeprelu(
    X: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    left_input: bool = True
) -> np.ndarray:
    input_ = X
    for idx, (w, b) in enumerate(zip(weights, biases)):
        operands = (input_, w)
        # tf uses left_input matmul
        # torch uses right_input matmul
        if not left_input:
            operands = list(reversed(operands))
            # torch 1d bias matrices inappropiate for transposed matmul sum
            b = np.expand_dims(b, 1)
        output_ = np.matmul(*operands) + b
        # add relu activations
        if idx > 0 and idx < (len(weights) - 1):
            output_ = np.vectorize(lambda x: np.max([x, 0]))(output_)
        input_ = output_
    return output_


def test_torch_network_representation():
    torch.manual_seed(1)
    torch_model = DeepReluNet(input_dim=1, L=2, M=2)
    torch_weights = set_torch_weights(torch_model, seed=1)

    X_test = np.array([[-1]])
    torch_loss, torch_pred = torch_eval(
        torch_model, torch.from_numpy(X_test).type(torch.FloatTensor))

    testing.assert_almost_equal(
        theoretical_deeprelu(
            X=np.array([[-1]]),
            weights=[w.detach().numpy() for w in torch_weights.values()][::2],
            biases=[w.detach().numpy() for w in torch_weights.values()][1::2],
            left_input=False),
        torch_pred.detach().numpy(),
        decimal=6  # works up to 7 decimals
    )


def test_tf_network_representation():
    # Careful with mixed precision policies in TF
    # The default policy in keras is 'float32'. Below examples from
    # https://www.tensorflow.org/guide/keras/mixed_precision
    # mixed_precision = tf.keras.mixed_precision.experimental
    # mixed_precision.set_policy(mixed_precision.Policy('mixed_float16'))
    # mixed_precision.set_policy(mixed_precision.Policy('float32'))
    clear_session()
    tf.random.set_seed(1)

    keras_model = deep_relu_net(input_dim=1, L=2, M=2)
    keras_weights = set_keras_weights(keras_model, seed=1)
    X_test = np.array([[-1]])
    keras_loss, keras_pred = keras_eval(keras_model, X_test)

    testing.assert_almost_equal(
        theoretical_deeprelu(
            X=np.array([[-1]]),
            weights=keras_weights[::2],
            biases=keras_weights[1::2],
            left_input=True),  # We have different operands order
        keras_pred,
        decimal=6  # works up to 6 decimals / not 7 like pytorch
    )
