import math

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


# We want to imitate the following KERAS model
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(64, activation='relu'))
#     model.add(layers.Dense(10))
#     return model


class MyDense(tf.Module):
    def __init__(
        self, in_features: int, out_features: int,
        with_relu: bool = True, name: str = "MyDense"
    ):
        super(MyDense, self).__init__(name=name)
        layer_dtype = tf.float64
        self.with_relu = with_relu

        stdv = 1. / math.sqrt(in_features)
        with tf.name_scope(name):
            self.w = tf.Variable(
                tf.random.uniform(
                    [in_features, out_features],
                    -stdv, stdv, dtype=layer_dtype),
                name="weights", dtype=layer_dtype)
            self.b = tf.Variable(
                tf.random.uniform(
                    [out_features], -stdv, stdv, dtype=layer_dtype),
                name="bias", dtype=layer_dtype)

    @tf.function
    def __call__(self, x, training=True):
        y = tf.matmul(x, self.w) + self.b
        if self.with_relu:
            return tf.nn.relu(y)
        return y


class MyConv2d(tf.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int,
        padding: Any = "VALID", strides: int = 1, with_relu: bool = True, name: str = "MyConv2d"
    ):
        super(MyConv2d, self).__init__(name=name)
        layer_dtype = tf.float64
        self.with_relu = with_relu
        self.padding = padding
        self.strides = strides

        in_features = in_channels * kernel_size * kernel_size
        stdv = 1. / math.sqrt(in_features)
        with tf.name_scope(name):
            self.w = tf.Variable(
                tf.random.uniform(
                    [kernel_size, kernel_size, in_channels, out_channels],
                    -stdv, stdv, dtype=layer_dtype),
                name="weights", dtype=layer_dtype)

            self.b = tf.Variable(
                tf.random.uniform(
                    [out_channels], -stdv, stdv, dtype=layer_dtype),
                name="bias", dtype=layer_dtype)

    @tf.function
    def __call__(self, x, training=True):
        y = tf.nn.conv2d(x, self.w, self.strides, self.padding) + self.b
        if self.with_relu:
            return tf.nn.relu(y)
        return y


class MyCNN(tf.Module):
    def __init__(self, name: str = "MyCNN"):
        super(MyCNN, self).__init__(name=name)
        self.dtype = tf.float64
        self.conv_layers = []
        filters = [3, 32, 64, 64]
        for i in range(3):
            self.conv_layers.append(
                MyConv2d(filters[i], filters[i + 1], 3, with_relu=True))

        self.lin_layers = []
        self.last_conv_dim = 64 * 4 * 4
        linear_size = 64
        num_classes = 10
        self.lin_layers.append(MyDense(self.last_conv_dim, linear_size, with_relu=True, name="fc1"))
        self.lin_layers.append(MyDense(linear_size, num_classes, with_relu=False, name="fc2"))

    @tf.function
    def __call__(self, x, training=True):
        next_input = x
        for lidx, layer in enumerate(self.conv_layers):
            next_input = layer(next_input)
            if lidx in [0, 1]:
                next_input = tf.nn.max_pool2d(next_input, ksize=2, strides=2, padding="VALID")

            dropout_probs = tf.expand_dims(
                tf.expand_dims(
                    tf.expand_dims(
                        tf.constant([0.95], dtype=self.dtype),
                        1), 1), 1) * tf.ones(tf.shape(next_input), dtype=self.dtype)
            mask = tfd.Bernoulli(probs=dropout_probs).sample()
            next_input = tf.cast(mask, dtype=self.dtype) * next_input

        next_input = tf.reshape(next_input, [tf.shape(x)[0], self.last_conv_dim])
        next_input = self.lin_layers[0](next_input)
        next_input = self.lin_layers[1](next_input)

        return next_input
