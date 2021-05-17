import math
from typing import Any

import tensorflow as tf

from self_tuning_nets.hyper.hyperparameters import HyperParameters

KERNEL_STDV = 0.1

class HyperConv2d(tf.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, hyper_params: HyperParameters,
        padding: Any = "VALID", strides: int = 1, with_relu: bool = True, name: str = "HyperConv2d"
    ):
        super(HyperConv2d, self).__init__(name=name)
        self.hyper_params = hyper_params
        hyper_dim = len(self.hyper_params.params)
        # layer setup
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
            self.hw = tf.Variable(
                tf.random.uniform(
                    [kernel_size, kernel_size, in_channels, out_channels],
                    -stdv, stdv, dtype=layer_dtype),
                name="hweights", dtype=layer_dtype)
            self.kw = tf.Variable(
                tf.random.normal(
                    [hyper_dim, out_channels], stddev=KERNEL_STDV, dtype=layer_dtype),
                name="hkweights", dtype=layer_dtype)

            # bias
            self.b = tf.Variable(
                tf.random.uniform(
                    [out_channels], -stdv, stdv, dtype=layer_dtype),
                name="bias", dtype=layer_dtype)
            self.hb = tf.Variable(
                tf.random.uniform(
                    [out_channels], -stdv, stdv, dtype=layer_dtype),
                name="hbias", dtype=layer_dtype)
            self.kb = tf.Variable(
                tf.random.normal(
                    [hyper_dim, out_channels], stddev=KERNEL_STDV, dtype=layer_dtype),
                name="hkbias", dtype=layer_dtype)

    @tf.function
    def __call__(self, x, training=True):
        hyper_unc_batch = self.hyper_params.all_params_unc_batch()
        oy = tf.nn.conv2d(x, self.w, self.strides, self.padding) + self.b
        hw = tf.expand_dims(tf.expand_dims(tf.linalg.matmul(hyper_unc_batch, self.kw), 1), 1)\
             * tf.nn.conv2d(x, self.hw, self.strides, self.padding)
        hb = tf.linalg.matmul(hyper_unc_batch, self.kb) * self.hb
        hb = tf.expand_dims(tf.expand_dims(hb, 1), 1)
        y = oy + hw + hb
        if self.with_relu:
            return tf.nn.relu(y)
        return y
