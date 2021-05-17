from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import datasets, layers, models
from tqdm.notebook import tqdm

from self_tuning_nets.hyper.custom_math import inv_softplus, s_logit, s_sigmoid
from self_tuning_nets.hyper.hyper_training import get_training_step_functions
from self_tuning_nets.hyper.hyperparameters import (HyperParameters,
                                                    hyper_param_from_bounds)
from self_tuning_nets.hyper.nets.hyper_custom_cnn import HyperCNN

tfd = tfp.distributions

# keras style imports
clear_session = tf.keras.backend.clear_session


@dataclass(frozen=True)
class ExperimentConfig:
    FRAMEWORK_SEED: int = 44
    INIT_DROPOUT: float = 0.95
    INIT_PIXEL_DROPOUT: float = 0.95
    INIT_AUGMENT_PROB: float = 0.05
    BATCH_SIZE: int = 32
    WARMUP_EPOCHS: int = 1
    MAX_EPOCHS: int = 20
    WEIGHT_STEPS: int = 2
    HYPER_STEPS: int = 1
    WITH_HYPER_TRAINING: bool = True


def run_deterministic_cpu_hyper_cnn_experiment(
    experiment_config: ExperimentConfig,
    for_hpo: bool = False
) -> List[np.ndarray]:
    """Here we implement an experiment with a simple custom CNN.

    The CNN hyperparameters include layer dropout probabilies
    The training dataset hyperparameters include the probability of
    using a data sample as a modified "augmented" sample and
    an image pixel dropout probability.
    
    We demonstrate how to reuse the same network to have a training
    without fine tuning the hyperparameters. Simply by disabling the
    hyperparameter perturbation and hyper_training step.
    """
    # Model initialization
    clear_session = tf.keras.backend.clear_session
    clear_session()
    tf.random.set_seed(experiment_config.FRAMEWORK_SEED)
    dtype = tf.float64

    # STEP 1: setup hyperparmeter variables
    hparams = [
        hyper_param_from_bounds(
        name=f"dropout{i}", init_val=experiment_config.INIT_DROPOUT,
        min_val=0.25, max_val=1.0) for i in range(3)
    ] + [
        hyper_param_from_bounds(
        name="pixel_drop", init_val=experiment_config.INIT_PIXEL_DROPOUT,
        min_val=0.25, max_val=1.0)
    ] + [
        hyper_param_from_bounds(
        name="augment_prob", init_val=experiment_config.INIT_AUGMENT_PROB,
        min_val=0.0, max_val=0.95)
    ]

    gen = tf.random.Generator.from_seed(experiment_config.FRAMEWORK_SEED)
    hyper_parameters = HyperParameters(hparams, experiment_config.BATCH_SIZE, gen)
    hvar_names = [v.name for v in hyper_parameters.get_all_hyperparameter_vars()]

    # STEP 2: setup model
    model = HyperCNN(hyper_parameters)

    # STEP 3: setup training
    opt_weights = tf.keras.optimizers.Adam()
    opt_hyper = tf.keras.optimizers.Adam(learning_rate=0.003)
    opt_scale = tf.keras.optimizers.Adam(learning_rate=0.003)
    weights_loss_obj = partial(tf.keras.losses.sparse_categorical_crossentropy, from_logits=True)
    hyperparam_loss_obj = partial(tf.keras.losses.sparse_categorical_crossentropy, from_logits=True)

    # STEP 4: setup datasets
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    dataset_size = 50000
    valid_samples_per_category = int(dataset_size * 0.2 // 10)  # 20% for 10 categories

    # Shuffle and separate training dataset into training and validation. 
    rng = np.random.default_rng(experiment_config.FRAMEWORK_SEED)
    validation_samples = np.concatenate([
        rng.choice(np.nonzero(train_labels == i)[0], valid_samples_per_category, replace=False)
        for i in range(10)], axis=0)
    training_samples = np.setdiff1d(
        np.array(range(dataset_size)), validation_samples, assume_unique=True)
    rng.shuffle(validation_samples)
    rng.shuffle(training_samples)
    
    # training dataset setup
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_images[training_samples], train_labels[training_samples]))
    train_ds = train_ds.shuffle(experiment_config.BATCH_SIZE)\
               .batch(experiment_config.BATCH_SIZE, drop_remainder=True).repeat()
    train_iter = iter(train_ds)

    def weights_data_func():
        X_train, Y_train = train_iter.get_next()
        X_train = tf.cast(X_train, dtype=dtype)
        pixel_drop = hyper_parameters.param_con_batch("pixel_drop")
        augment_prob = hyper_parameters.param_con_batch("augment_prob")
        augmented = tf.cast(tfd.Bernoulli(probs=augment_prob).sample(), dtype=tf.bool)
        new_images = tf.TensorArray(dtype, size=experiment_config.BATCH_SIZE)
        for idx in tf.range(tf.constant(experiment_config.BATCH_SIZE)):
            image = X_train[idx]
            if augmented[idx]:
                image = image * tf.cast(
                    tfd.Bernoulli(probs=pixel_drop[idx][0]).sample(tf.shape(image)), dtype=dtype)
            new_images = new_images.write(idx, image)
        return new_images.stack(), Y_train

    # validation dataset setup
    valid_ds = tf.data.Dataset.from_tensor_slices(
        (train_images[validation_samples], train_labels[validation_samples]))
    valid_ds = valid_ds.shuffle(experiment_config.BATCH_SIZE)\
               .batch(experiment_config.BATCH_SIZE, drop_remainder=True).repeat()
    valid_iter = iter(valid_ds)

    def hyper_data_func():
        X_train, Y_train = valid_iter.get_next()
        return tf.cast(X_train, dtype=dtype), Y_train

    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.batch(experiment_config.BATCH_SIZE, drop_remainder=True).repeat()
    test_iter = iter(test_ds)

    def test_data_func():
        X_train, Y_train = test_iter.get_next()
        return tf.cast(X_train, dtype=dtype), Y_train

    # Prepare metrics
    wlosses = []
    hlosses = []
    param_trajectories = {h.name: [] for h in hparams}
    scale_trajectories = {h.name: [] for h in hparams}
    accuracy = []
    acc = tf.keras.metrics.SparseCategoricalAccuracy()

    # STEP 5: Setup training loop
    # warm up
    steps_per_epoch = 40000 // experiment_config.BATCH_SIZE
    warmup_steps = experiment_config.WARMUP_EPOCHS * steps_per_epoch
    weights_training_step, hyperparameters_training_step = get_training_step_functions()
    for step in tqdm(range(warmup_steps),
                     total=warmup_steps, unit="warm_up_batch"):
        weights_training_step(
            model, hyper_parameters, weights_data_func,
            weights_loss_obj, opt_weights, experiment_config.WITH_HYPER_TRAINING
        )

    # training
    steps_per_epoch = 40000 // (experiment_config.BATCH_SIZE * experiment_config.WEIGHT_STEPS)
    training_steps = experiment_config.MAX_EPOCHS * steps_per_epoch
    for step in tqdm(range(training_steps),
                     total=training_steps, unit="training_cycle"):
        for _ in range(experiment_config.WEIGHT_STEPS):
            wloss = weights_training_step(
                model, hyper_parameters, weights_data_func,
                weights_loss_obj, opt_weights, experiment_config.WITH_HYPER_TRAINING
            )
        if experiment_config.WITH_HYPER_TRAINING:
            for _ in range(experiment_config.HYPER_STEPS):
                hloss = hyperparameters_training_step(
                    model, hyper_parameters, hyper_data_func,
                    hyperparam_loss_obj, opt_hyper, opt_scale
                )

        # record metrics eagerly
        if step % steps_per_epoch == 0:
            test_steps = 10000 // experiment_config.BATCH_SIZE
            hyper_parameters.disable_perturbation()
            acc.reset_states()
            for _ in range(test_steps):
                X_test, Y_test = test_data_func()
                Y_pred = model(X_test, training=False)
                acc.update_state(Y_test, Y_pred)
            accuracy.append(acc.result().numpy())
            wlosses.append(wloss.numpy())
            # hyper metrics
            hyper_vars = hyper_parameters.hyper_param_vars
            for h in hparams:
                param_trajectories[h.name].append(h.unc_to_con(hyper_vars[h.name][0]).numpy())
                scale_trajectories[h.name].append(tf.math.softplus(hyper_vars[h.name][1]).numpy())
            if experiment_config.WITH_HYPER_TRAINING:
                hlosses.append(hloss.numpy())

    # Return validation accuracy for external hyper parameter optimization
    if for_hpo:
        test_steps = 10000 // experiment_config.BATCH_SIZE
        hyper_parameters.disable_perturbation()
        acc.reset_states()
        for _ in range(test_steps):
            X_test, Y_test = hyper_data_func()
            Y_pred = model(X_test, training=False)
            acc.update_state(Y_test, Y_pred)
        hyper_accuracy = acc.result().numpy()
        return wlosses, hlosses, param_trajectories, scale_trajectories, accuracy, hyper_accuracy

    return wlosses, hlosses, param_trajectories, scale_trajectories, accuracy
