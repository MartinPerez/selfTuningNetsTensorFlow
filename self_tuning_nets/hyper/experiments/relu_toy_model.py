from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List

import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm

from self_tuning_nets.hyper.custom_math import inv_softplus, s_logit, s_sigmoid
from self_tuning_nets.hyper.hyper_training import get_training_step_functions
from self_tuning_nets.hyper.hyperparameters import (HyperParameters,
                                                    hyper_param_from_bounds)
from self_tuning_nets.hyper.nets.hyper_relu_net import HyperReluNet

# keras style imports
clear_session = tf.keras.backend.clear_session


@dataclass(frozen=True)
class ExperimentConfig:
    MODEL_L: int = 2
    MODEL_M: int = 2
    FRAMEWORK_SEED: int = 42
    BATCH_SIZE: int = 100
    INIT_TRAIN_RANGE: float = 1.0
    EVAL_SIZE: int = 100
    EVAL_RANGE: float = 2.0
    PRED_SAMPLING_STEP: int = 50
    MAX_TRAINING_CYCLES: int = 2000
    WARMUP_STEPS = 5
    WEIGHT_STEPS = 2
    HYPER_STEPS = 2


def run_deterministic_cpu_hyper_relu_experiment(
    experiment_config: ExperimentConfig,
    verbose: int = 0
) -> List[np.ndarray]:
    """Here we implement with a dummy model the self tuning networks of
       MacKay et al. available in https://arxiv.org/pdf/1903.03088.pdf.

       With the dummy model we can demonstrate the converging hyperparameter
       trajectories in contrast with the fixed hyperparameter behavior of
       the linear experiments in linear_experiment.py.

       A notebook demonstration is given in notebooks/tf_hypertraining.ipynb
    """
    # Deterministic behavior
    clear_session = tf.keras.backend.clear_session
    clear_session()
    tf.random.set_seed(experiment_config.FRAMEWORK_SEED)

    # STEP 1: setup hyperparmeters
    x_range = hyper_param_from_bounds(
        name="x_range", init_val=experiment_config.INIT_TRAIN_RANGE,
        min_val=0.01, max_val=10.0)
    gen = tf.random.Generator.from_seed(experiment_config.FRAMEWORK_SEED)
    hyper_parameters = HyperParameters([x_range], experiment_config.BATCH_SIZE, gen)

    # STEP 2: setup model
    model = HyperReluNet(experiment_config.MODEL_L, experiment_config.MODEL_M, hyper_parameters)

    # STEP 3: setup training objects
    opt_weights = tf.keras.optimizers.Adam(learning_rate=0.01)
    opt_hyper = tf.keras.optimizers.Adam(learning_rate=0.003)
    opt_scale = tf.keras.optimizers.Adam(learning_rate=0.003)
    weights_loss_obj = tf.keras.losses.MSE
    hyperparam_loss_obj = tf.keras.losses.MSE

    # STEP 4: setup data functions
    def weights_training_x_interval_batch(hyper_parameters):
        con_x_range_batch = hyper_parameters.param_con_batch("x_range")
        x_sample = gen.uniform(
            shape=(hyper_parameters.batch_size, 1), minval=-1, maxval=1, dtype=tf.float64)
        scaled_sample = tf.multiply(con_x_range_batch, x_sample)
        return scaled_sample, scaled_sample ** tf.constant(2.0, dtype=tf.float64)

    weights_data_func = partial(weights_training_x_interval_batch, hyper_parameters)

    def hyper_data_func():
        X_train = gen.uniform(
            shape=(experiment_config.BATCH_SIZE, 1),
            minval=-experiment_config.EVAL_RANGE,
            maxval=experiment_config.EVAL_RANGE,
            dtype=tf.float64)
        Y_train = X_train ** tf.constant(2.0, dtype=tf.float64)
        return X_train, Y_train

    # STEP 5: setup training loop with metrics
    X_eval = tf.constant(
        np.expand_dims(
            np.linspace(
                -experiment_config.EVAL_RANGE,
                experiment_config.EVAL_RANGE,
                100),
            1),
        dtype=tf.float64)
    Y_eval = X_eval ** tf.constant(2.0, dtype=tf.float64)

    # Capture initial function
    predictions = []
    pred = model(X_eval, training=False)
    predictions.append(pred)

    # Capture function trajectory
    x_range_trajectory = []
    x_scaling_trajectory = []
    dist_trajectory = []
    wlosses = []
    hlosses = []

    weights_training_step, hyperparameters_training_step = get_training_step_functions()
    for _ in range(experiment_config.WARMUP_STEPS):
        weights_training_step(
            model, hyper_parameters, weights_data_func,
            weights_loss_obj, opt_weights
        )

    for step in tqdm(range(experiment_config.MAX_TRAINING_CYCLES),
                     total=experiment_config.MAX_TRAINING_CYCLES, unit="cycle"):
        for _ in range(experiment_config.WEIGHT_STEPS):
            wloss = weights_training_step(
                model, hyper_parameters, weights_data_func,
                weights_loss_obj, opt_weights
            )
        for _ in range(experiment_config.HYPER_STEPS):
            hloss = hyperparameters_training_step(
                model, hyper_parameters, hyper_data_func,
                hyperparam_loss_obj, opt_hyper, opt_scale
            )

        # # eval and metrics
        Y_pred = model(X_eval, training=False)
        dist = float(np.max(np.abs(Y_eval.numpy() - Y_pred.numpy())))
        dist_trajectory.append(dist)

        hyper_vars = hyper_parameters.hyper_param_vars
        x_range_trajectory.append(x_range.unc_to_con(hyper_vars["x_range"][0]).numpy())
        x_scaling_trajectory.append(tf.math.softplus(hyper_vars["x_range"][1]).numpy())

        wlosses.append(wloss.numpy())
        hlosses.append(hloss.numpy())

        if dist < 0.05:
            break
        if step % experiment_config.PRED_SAMPLING_STEP == 0:
            predictions.append(Y_pred.numpy())
            if verbose > 0:
                print("dist: ", dist)
    if step % experiment_config.PRED_SAMPLING_STEP != 0:
        predictions.append(Y_pred.numpy())
        if verbose > 0:
            print("dist: ", dist)

    return predictions, x_range_trajectory, x_scaling_trajectory, \
        dist_trajectory, wlosses, hlosses
