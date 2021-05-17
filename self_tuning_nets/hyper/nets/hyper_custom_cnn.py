import tensorflow as tf

from self_tuning_nets.hyper.hyperparameters import HyperParameters
from self_tuning_nets.hyper.layers.dropout2d import dropout2d_layer
from self_tuning_nets.hyper.layers.hyper_conv2d import HyperConv2d
from self_tuning_nets.hyper.layers.hyper_dense import HyperDense


# We want to imitate the following KERAS model
"""
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.05))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.05))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.05))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    return model
"""

class HyperCNN(tf.Module):
    def __init__(self, hyper_params: HyperParameters, name: str = "HyperCNN"):
        super(HyperCNN, self).__init__(name=name)
        self.hyper_params = hyper_params

        self.conv_layers = []
        filters = [3, 32, 64, 64]
        for i in range(3):
            self.conv_layers.append(
                HyperConv2d(filters[i], filters[i + 1], 3, hyper_params, with_relu=True))

        self.lin_layers = []
        self.last_conv_dim = 64 * 4 * 4
        linear_size = 64
        num_classes = 10
        self.lin_layers.append(HyperDense(self.last_conv_dim, linear_size, hyper_params, with_relu=True, name="fc1"))
        self.lin_layers.append(HyperDense(linear_size, num_classes, hyper_params, with_relu=False, name="fc2"))

    @tf.function
    def __call__(self, x, training=True):
        next_input = x
        for lidx, layer in enumerate(self.conv_layers):
            next_input = layer(next_input, training)
            if lidx in [0, 1]:
                next_input = tf.nn.max_pool2d(next_input, ksize=2, strides=2, padding="VALID")
            if training:
                next_input = dropout2d_layer(next_input, self.hyper_params.param_con_batch(f"dropout{lidx}"))

        next_input = tf.reshape(next_input, (tf.shape(x)[0], self.last_conv_dim))
        next_input = self.lin_layers[0](next_input, training)
        next_input = self.lin_layers[1](next_input, training)

        return next_input
