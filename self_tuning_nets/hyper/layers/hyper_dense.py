import math

import tensorflow as tf

from self_tuning_nets.hyper.hyperparameters import HyperParameters

KERNEL_STDV = 0.1

class HyperDense(tf.Module):
    def __init__(
        self, in_features: int, out_features: int, hyper_params: HyperParameters,
        with_relu: bool = True, name: str = "HyperDense"
    ):
        super(HyperDense, self).__init__(name=name)
        self.hyper_params = hyper_params
        hyper_dim = len(self.hyper_params.params)
        # layer setup
        layer_dtype = tf.float64
        self.with_relu = with_relu

        stdv = 1. / math.sqrt(in_features)
        with tf.name_scope(name):
            self.w = tf.Variable(
                tf.random.uniform(
                    [in_features, out_features],
                    -stdv, stdv, dtype=layer_dtype),
                name="weights", dtype=layer_dtype)
            self.hw = tf.Variable(
                tf.random.uniform(
                    [in_features, out_features],
                    -stdv, stdv, dtype=layer_dtype),
                name="hweights", dtype=layer_dtype)
            self.kw = tf.Variable(
                tf.random.normal(
                    [hyper_dim, 1], stddev=KERNEL_STDV, dtype=layer_dtype),
                name="hkweights", dtype=layer_dtype)
            self.b = tf.Variable(
                tf.random.uniform(
                    [out_features], -stdv, stdv, dtype=layer_dtype),
                name="bias", dtype=layer_dtype)
            self.hb = tf.Variable(
                tf.random.uniform(
                    [out_features], -stdv, stdv, dtype=layer_dtype),
                name="hbias", dtype=layer_dtype)
            self.kb = tf.Variable(
                tf.random.normal(
                    [hyper_dim, 1], stddev=KERNEL_STDV, dtype=layer_dtype),
                name="hkbias", dtype=layer_dtype)

    @tf.function
    def __call__(self, x, training=True):
        hyper_unc_batch = self.hyper_params.all_params_unc_batch()
        oy = tf.matmul(x, self.w) + self.b
        hw = tf.linalg.matmul(hyper_unc_batch, self.kw) * tf.matmul(x, self.hw)
        hb = tf.linalg.matmul(hyper_unc_batch, self.kb) * self.hb
        y = oy + hw + hb
        if self.with_relu:
            return tf.nn.relu(y)
        return y
