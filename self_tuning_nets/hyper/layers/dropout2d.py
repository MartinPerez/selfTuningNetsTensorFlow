import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def dropout2d_layer(
    next_input: tf.Tensor,
    con_dropout_batch: tf.Tensor,
    dtype: tf.DType = tf.float64
) -> tf.Tensor:
    expanded_tensor = con_dropout_batch
    for _ in range(len(tf.shape(next_input)) - 2):
        expanded_tensor = tf.expand_dims(expanded_tensor, 1)
    dropout_probs = (expanded_tensor *
                     tf.ones(tf.shape(next_input), dtype=dtype))
    mask = tfd.Bernoulli(probs=dropout_probs).sample()
    return tf.cast(mask, dtype=dtype) * next_input
