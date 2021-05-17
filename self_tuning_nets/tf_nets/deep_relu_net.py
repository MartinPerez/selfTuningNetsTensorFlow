import tensorflow as tf

# Its very important for deterministic behavior in TF 2.0 that we
# use the internal tf keras and not the external package

# keras style imports
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Activation = tf.keras.layers.Activation
Adam = tf.keras.optimizers.Adam


def deep_relu_net(input_dim: int, L: int, M: int) -> tf.keras.models.Model:
    depth = L + 1
    width = M + input_dim + 1
    model = Sequential()

    model.add(Dense(width, input_shape=(input_dim,), activation=None))
    for _ in range(depth):
        model.add(Dense(width, activation='relu'))
    model.add(Dense(1, activation=None))
    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss='mse',
                  metrics=[])
    return model
