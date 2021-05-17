import tensorflow as tf

from self_tuning_nets.hyper.hyperparameters import HyperParameters
from self_tuning_nets.hyper.layers.dropout2d import dropout2d_layer
from self_tuning_nets.hyper.layers.hyper_conv2d import HyperConv2d
from self_tuning_nets.hyper.layers.hyper_dense import HyperDense


class HyperAlexNet(tf.Module):
    def __init__(self, hyper_params: HyperParameters, name: str = "HyperAlexNet"):
        super(HyperAlexNet, self).__init__(name=name)
        self.hyper_params = hyper_params
        self.conv_layers = []
        filters = [3, 64, 192, 384, 256, 256]
        strides = [2, 1, 1, 1, 1]
        padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
        conv_depth = 5
        for i in range(conv_depth):
            self.conv_layers.append(
                HyperConv2d(filters[i], filters[i + 1], 3, hyper_params,
                            strides=strides[i], padding=padding))

        self.lin_layers = []
        self.last_dim = 256 * 2 * 2
        linear_size = 4096
        num_classes = 10
        self.lin_layers.append(HyperDense(self.last_dim, linear_size, hyper_params, with_relu=True, name="fc1"))
        self.lin_layers.append(HyperDense(linear_size, linear_size, hyper_params, with_relu=True, name="fc2"))
        self.lin_layers.append(HyperDense(linear_size, num_classes, hyper_params, with_relu=False, name="fc3"))     

    @tf.function
    def __call__(self, x, training=True):
        next_input = x
        for lidx, layer in enumerate(self.conv_layers):
            next_input = layer(next_input, training)
            if lidx in [0, 1, 4]:
                next_input = tf.nn.max_pool2d(next_input, ksize=2, strides=2, padding="VALID")
            if training:
                next_input = dropout2d_layer(next_input, self.hyper_params.param_con_batch(f"dropout{lidx}"))

        next_input = tf.reshape(next_input, (tf.shape(x)[0], self.last_dim))
        next_input = dropout2d_layer(next_input, self.hyper_params.param_con_batch(f"dropout5"))
        next_input = self.lin_layers[0](next_input, training)
        next_input = dropout2d_layer(next_input, self.hyper_params.param_con_batch(f"dropout6"))
        next_input = self.lin_layers[1](next_input, training)
        next_input = self.lin_layers[2](next_input, training)
        return next_input
