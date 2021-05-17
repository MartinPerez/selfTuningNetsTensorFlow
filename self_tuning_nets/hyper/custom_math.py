import tensorflow as tf


def logit(x, dtype=tf.float64):
    return tf.math.log(x) - tf.math.log(tf.constant(1.0, dtype=dtype) - x)


def s_logit(x, min_val, max_val, dtype=tf.float64):
    return logit((x - min_val)/(max_val-min_val), dtype)


def inv_softplus(x, dtype=tf.float64):
    return tf.math.log(tf.math.exp(x) - tf.constant(1.0, dtype=dtype))


def s_sigmoid(unc_hyperparam, min_val, max_val, dtype=tf.float64):
    return (max_val - min_val) * tf.math.sigmoid(unc_hyperparam) + min_val
