from typing import Any, Callable, List

import tensorflow as tf

from self_tuning_nets.hyper.hyperparameters import HyperParameters

# Open issue to pass optimizer
# https://github.com/tensorflow/tensorflow/issues/27120
# Need to wrapt tf.function to redefine them across experiments

def get_training_step_functions():
    """ Provide the weight training and hyperparameter training steps for customization
    
    The weights training step only calls hyperparameter perturbations if hyper_training
    is allowed. Like this we can easily train a model with fixed hyperparameters as well
    as using the self-tuning approach.

    We need to provide the weights training step with:
        - model (TF module): The network model
        - hyper_parameters (HyperParameters): The HyperParameters module
        - data_func (Callable): Provides (X, y) batch tensors and takes no arguments
        - loss_obj (Any): TF API loss function that takes X, y as arguments
        - opt_weights_obj (Any): Tf API optimizer object with apply_gradients method
        - with_hyper_training (bool): flag to train with fixed hyperparameter values

    The hyperparameters training step optimizes the hyperparameter values and the
    scale of perturbations of the respective hyperparameters.

    We need to provide the hyperparameters training step with:
        - model (TF module): The network model
        - hyper_parameters (HyperParameters): The HyperParameters module
        - data_func (Callable): Provides (X, y) batch tensors and takes no arguments
        - loss_obj (Any): TF API loss function that takes X, y as arguments
        - opt_hyper_obj (Any): Tf API optimizer object with apply_gradients method
        - opt_scale_obj (Any): Tf API optimizer object with apply_gradients method

    Returns:
    weights_training_step (tf.Function): updates model weights
    hyperparameters_training_step (tf.Function): updates hyperparameter related variables 
    """
    @tf.function
    def weights_training_step(
        model: tf.Module,
        hyper_parameters: HyperParameters,
        data_func: Callable,
        loss_obj: Any,
        opt_weights_obj: Any,
        with_hyper_training: bool = True
    ) -> tf.Tensor:
        if tf.constant(with_hyper_training, dtype=tf.bool):
            hyper_parameters.perturb()
        X_train, Y_train = data_func()

        # set tape loss
        weights = hyper_parameters.get_model_vars(model)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(weights)
            Y_pred = model(X_train, training=True)
            loss = loss_obj(Y_train, Y_pred)

        # update gradient
        grads = tape.gradient(loss, weights)
        opt_weights_obj.apply_gradients(zip(grads, weights))
        return tf.math.reduce_mean(loss)


    @tf.function
    def hyperparameters_training_step(
        model: tf.Module,
        hyper_parameters: HyperParameters,
        data_func: Callable,
        loss_obj: Any,
        opt_hyper_obj: Any,
        opt_scale_obj: Any
    ) -> tf.Tensor:
        hyper_parameters.perturb()
        X_train, Y_train = data_func()

        # Hyperparam update
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(hyper_parameters.get_unc_vars())
            Y_pred = model(X_train, training=False)
            loss = loss_obj(Y_train, Y_pred) - hyper_parameters.hyper_entropy()
        grad = tape.gradient(loss, hyper_parameters.get_unc_vars())
        opt_hyper_obj.apply_gradients(zip(grad, hyper_parameters.get_unc_vars()))

        # Perturbation scale update
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(hyper_parameters.get_scale_vars())
            Y_pred = model(X_train, training=False)
            loss = loss_obj(Y_train, Y_pred) - hyper_parameters.hyper_entropy()
        grad = tape.gradient(loss, hyper_parameters.get_scale_vars())
        opt_scale_obj.apply_gradients(zip(grad, hyper_parameters.get_scale_vars()))
        return tf.math.reduce_mean(loss)

    return weights_training_step, hyperparameters_training_step
