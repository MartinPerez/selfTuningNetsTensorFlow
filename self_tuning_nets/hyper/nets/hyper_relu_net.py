import math

import tensorflow as tf

from self_tuning_nets.hyper.hyperparameters import HyperParameters
from self_tuning_nets.hyper.layers.hyper_dense import HyperDense


class HyperReluNet(tf.Module):
    def __init__(self, L: int, M: int, hyper_params: HyperParameters,
                 name: str = "HyperReluNet"):
        super(HyperReluNet, self).__init__(name=name)
        # model setup
        input_dim = 1
        depth = L + 1
        width = M + input_dim + 1

        self.layers = []
        self.layers.append(HyperDense(
            input_dim, width, hyper_params, with_relu=False, name="dense_input"))
        for i in range(depth):
            self.layers.append(HyperDense(
                width, width, hyper_params, with_relu=True, name=(f"hidden_{i}")))
        self.layers.append(HyperDense(
            width, 1, hyper_params, with_relu=False, name="dense_output"))

    @tf.function
    def __call__(self, x, training=True):
        next_input = x
        for layer in self.layers:
            next_input = layer(next_input, training)
        return next_input
