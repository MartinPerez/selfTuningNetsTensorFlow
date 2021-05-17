from dataclasses import dataclass
from typing import List

import numpy as np
import tensorflow as tf
import torch
from tqdm.notebook import tqdm

from self_tuning_nets.dataset import x_interval_batches
from self_tuning_nets.tf_nets.deep_relu_net import deep_relu_net
from self_tuning_nets.tf_nets.training import eval_step as keras_eval
from self_tuning_nets.tf_nets.training import set_weights as set_keras_weights
from self_tuning_nets.tf_nets.training import train_step as keras_train
from self_tuning_nets.torch_nets.deep_relu_net import DeepReluNet
from self_tuning_nets.torch_nets.training import eval_step as torch_eval
from self_tuning_nets.torch_nets.training import \
    set_weights as set_torch_weights
from self_tuning_nets.torch_nets.training import train_step as torch_train

# keras style imports
clear_session = tf.keras.backend.clear_session


def theoretical_bounds_metric(
    y_train: np.ndarray,
    y_pred: np.ndarray,
    L: int
) -> float:
    # Theoretical distance used by Weinan et al in
    # https://arxiv.org/pdf/1807.00297.pdf
    assert len(y_train.shape) == 2  # shape (n, 1) expected
    assert y_train.shape == y_pred.shape
    return float(np.max(np.abs(y_train - y_pred)) - pow(2, -2 * L))


@dataclass
class ExperimentConfig:
    WEIGHTS_SEED: int = 42
    DATA_SEED: int = 42
    FRAMEWORK_SEED: int = 42
    TRAIN_RANGE: float = 1.0
    BATCH_SIZE: int = 100
    MAX_BATCHES: int = 10000
    EVAL_SIZE: int = 100
    EVAL_RANGE: float = 1
    MODEL_L: int = 2
    MODEL_M: int = 2
    PRED_SAMPLING_STEP: int = 50


def run_deterministic_cpu_basic_torch_experiment(
    experiment_config: ExperimentConfig,
    X_eval: np.ndarray,  # shape (n, 1) expected
    verbose: int = 0
) -> List[np.ndarray]:
    """This function does not control randomness in a GPU setting:
    https://pytorch.org/docs/stable/notes/randomness.html

    for gpu considerations we need to also explore
    torch.cuda.manual_seed_all(n)
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed(n)
    for numpy consideration
    np.random.seed(0)
    """
    # Model initialization
    torch.manual_seed(experiment_config.FRAMEWORK_SEED)

    model = DeepReluNet(
        input_dim=1,
        L=experiment_config.MODEL_L,
        M=experiment_config.MODEL_M)

    set_torch_weights(model, experiment_config.WEIGHTS_SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data_iter = x_interval_batches(
        iter_seed=experiment_config.DATA_SEED,
        max_batches=experiment_config.MAX_BATCHES,
        batch_size=experiment_config.BATCH_SIZE,
        x_bound=experiment_config.TRAIN_RANGE)

    # Capture initial function
    predictions = []
    X_eval_tensor = torch.from_numpy(X_eval)\
        .type(torch.FloatTensor)
    _, pred = torch_eval(model, X_eval_tensor)
    predictions.append(pred.detach().numpy())

    # Capture function trajectory
    for step, X_train in tqdm(enumerate(data_iter),
                              total=experiment_config.MAX_BATCHES,
                              unit="batch"):
        # shape (n, 1) expected
        X_train = torch.from_numpy(X_train).unsqueeze_(1)
        model, t_loss = torch_train(optimizer, model, X_train)
        e_loss, pred = torch_eval(model, X_eval_tensor)

        dist = theoretical_bounds_metric(
            X_eval ** 2,
            pred.detach().numpy(),
            experiment_config.MODEL_L)
        if dist < 0.0:
            break
        if step % experiment_config.PRED_SAMPLING_STEP == 0:
            predictions.append(pred.detach().numpy())
            if verbose > 0:
                print(t_loss.detach().numpy(),
                      e_loss.detach().numpy(),
                      dist)
    if verbose > 0:
        print(t_loss.detach().numpy(),
              e_loss.detach().numpy(),
              dist)

    # Capture last function if not already considered
    if step % experiment_config.PRED_SAMPLING_STEP != 0:
        predictions.append(pred.detach().numpy())
    return predictions


def run_deterministic_cpu_basic_keras_experiment(
    experiment_config: ExperimentConfig,
    X_eval: np.ndarray,  # shape (n, 1) expected
    verbose: int = 0
) -> List[np.ndarray]:
    # Model initialization
    # tf.config.set_visible_devices([], 'GPU') ? could remove gpu visibility
    # os.environ['TF_DETERMINISTIC_OPS'] = "1" ? set_seed seems enough
    clear_session()
    # sets keras randomization behavior but does not impact adam steps
    tf.random.set_seed(experiment_config.FRAMEWORK_SEED)

    model = deep_relu_net(
        input_dim=1,
        L=experiment_config.MODEL_L,
        M=experiment_config.MODEL_M)
    if verbose > 0:
        model.summary()

    # Control randomization
    set_keras_weights(model, experiment_config.WEIGHTS_SEED)
    data_iter = x_interval_batches(
        iter_seed=experiment_config.DATA_SEED,
        max_batches=experiment_config.MAX_BATCHES,
        batch_size=experiment_config.BATCH_SIZE,
        x_bound=experiment_config.TRAIN_RANGE)

    # Capture initial function
    predictions = []
    _, pred = keras_eval(model, X_eval)
    predictions.append(pred)

    # Capture function trajectory
    for step, X_train in tqdm(enumerate(data_iter),
                              total=experiment_config.MAX_BATCHES,
                              unit="batch"):
        # shape (n, 1) expected
        model, t_loss = keras_train(model, X_train)
        e_loss, pred = keras_eval(model, X_eval)

        dist = theoretical_bounds_metric(
            X_eval ** 2,
            pred,
            experiment_config.MODEL_L)
        if dist < 0.0:
            break
        if step % experiment_config.PRED_SAMPLING_STEP == 0:
            predictions.append(pred)
            if verbose > 0:
                print(t_loss, e_loss, dist)
    if step % experiment_config.PRED_SAMPLING_STEP != 0:
        predictions.append(pred)
        if verbose > 0:
            print(t_loss, e_loss, dist)

    return predictions
