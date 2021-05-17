import math
from dataclasses import dataclass
from typing import List

import tensorflow as tf

from self_tuning_nets.hyper.custom_math import (inv_softplus, s_logit,
                                                  s_sigmoid)


@dataclass
class HyperParam:
    """ Represents the hyperparameter definition
    
    A hyperparameter exists in two invertible spaces:
        * The constrained space where the semantic value is defined, like a probability
        * The unconstrained space where the gradient update is performed
    We need the user to provide a reasonable implementation of this invertible
    mapping for a given hyperparameter.
    Moreover we need an initial value for the scale of perturbations of the
    hyperparameter during training. We propose from the literature to keep at 0.5

    Parameters:
    name (str): Unique identifier for the parameter
    unc_init (TF eager Function): Provides init value tensor in unconstrained space
    unc_to_con (TF eager Function): Provides map from unconstrained to constrained space
    scale_init_val (tf.Tensor): init value for the scale of training perturbations
    """
    name: str
    unc_init: tf.python.eager.def_function.Function
    unc_to_con: tf.python.eager.def_function.Function
    scale_init_val: tf.Tensor = tf.constant(0.5, dtype=tf.float64)


def hyper_param_from_bounds(
    name: str,
    init_val: float,
    min_val: float,
    max_val: float,
    dtype: tf.DType = tf.float64
) -> HyperParam:
    """ Utility function to define a continous low and high bounded hyperparameter

    Parameters:
    name (str): Unique identifier for the parameter
    init_val (float): Initial value
    min_val (float): Minimum value
    max_val (float): Maximum value
    dtype (tf.Dtype): Tensors dtype, default=tf.float64

    Returns:
    HyperParam: suggested hyperparameter initialization and invertible mapping
    """
    if init_val < min_val or init_val > max_val:
        raise Exception(f"{name} init val {init_val} not between {min_val} and {max_val}")

    def unc_init():
        hinit = tf.constant(init_val, dtype=dtype)
        minv = tf.constant(min_val, dtype=dtype)
        maxv = tf.constant(max_val, dtype=dtype)
        return s_logit(hinit, minv, maxv)

    def unc_to_con(unc_tensor):
        minv = tf.constant(min_val, dtype=dtype)
        maxv = tf.constant(max_val, dtype=dtype)
        return s_sigmoid(unc_tensor, minv, maxv)

    return HyperParam(name, unc_init, unc_to_con)


def _entropy_loss(hscale, dtype=tf.float64):
    scale = tf.math.softplus(hscale)
    return tf.math.log(scale * tf.math.sqrt(
        tf.constant(2.0 * math.pi * math.e, dtype=dtype)))


class HyperParameters(tf.Module):
    """Hyperparameters provider for training network models and data preprocessing

    This class handles several important functions for the self-tuning networks approach
        - Its a provider for batched perturbed constrained space parameter tensors
        - Wires up the variables required to propagate gradiets to hyperparameters
        - Utility functions for training steps to gather Tensor Variables
        - Utility function to disable hyperparameter training
    
    For example param_con_batch(param_name) allows to obtain the batched and
    perturbed hyperparameter values to use the tensor for batch data provider function
    or when wiring up a network model module in its __call__ method. The HyperParameters
    object should be provided to a network during initialization to be referenced
    during graph construction time in the __call__ method.

    Parameters:
    params (List[HyperParam]): List of hyperparameter definitions
    batch_size (int): size of hyperparameter batched tensors
    gen (tf.random.Generator): Generator to control seeding of perturbations
    dtype (tf.Dtype): dtype of tensor variables to be instantiated for hyperparameters
    name (str): Module name. Default=HyperParameters
    """
    def __init__(self,
                params: List[HyperParam],
                batch_size: int,
                gen: tf.random.Generator,
                dtype: tf.DType = tf.float64,
                name: str='HyperParameters'):
        super(HyperParameters, self).__init__(name=name)
        self.dtype = dtype
        self.params = params
        self.batch_size = batch_size
        self.gen = gen
        self.hyper_param_vars = {}
        self.con_mappings = {}
        with tf.name_scope(name) as scope:
            for param in params:
                unc_var = tf.Variable(
                    param.unc_init(), name=f"{param.name}_unc", dtype=self.dtype)
                scale_var = tf.Variable(
                    inv_softplus(param.scale_init_val),
                    name=f"{param.name}_scale", dtype=self.dtype)
                perturb_var = tf.Variable(
                    self.gen.normal(shape=(self.batch_size, 1), dtype=self.dtype),
                    name=f"{param.name}_perturb", dtype=self.dtype, trainable=False)
                self.hyper_param_vars[param.name] = (unc_var, scale_var, perturb_var)
                self.con_mappings[param.name] = param.unc_to_con
            self.entropy_coefficient = tf.constant(0.001, dtype=self.dtype)

    def param_unc_batch(self, unc_var, scale_var, perturb_var):
        unc_repeated = tf.reshape(
            tf.repeat(unc_var, repeats=[self.batch_size]),
            (self.batch_size, 1))
        return unc_repeated + tf.math.softplus(scale_var) * perturb_var

    def all_params_unc_batch(self):
        return tf.concat(
            [self.param_unc_batch(*self.hyper_param_vars[param.name])
             for param in self.params],
            1)

    def param_con_batch(self, name):
        unc_batch = self.param_unc_batch(*self.hyper_param_vars[name])
        return self.con_mappings[name](unc_batch)

    def perturb(self):
        for param in self.params:
            perturb_var = self.hyper_param_vars[param.name][2]
            perturb_var.assign(self.gen.normal(shape=(self.batch_size, 1), dtype=self.dtype))

    def hyper_entropy(self):
        return self.entropy_coefficient * tf.math.add_n(
            [_entropy_loss(v[1], self.dtype) for v in self.hyper_param_vars.values()])

    def get_unc_vars(self):
        return [self.hyper_param_vars[param.name][0] for param in self.params]

    def get_scale_vars(self):
        return [self.hyper_param_vars[param.name][1] for param in self.params]

    def get_all_hyperparameter_vars(self):
        return [v for param in self.params for v in self.hyper_param_vars[param.name]]

    def get_model_vars(self, model):
        hvar_names = [v.name for v in self.get_all_hyperparameter_vars()]
        return [v for v in model.variables if v.name not in hvar_names]

    def disable_perturbation(self):
        for param in self.params:
            perturb_var = self.hyper_param_vars[param.name][2]
            perturb_var.assign(tf.zeros(shape=(self.batch_size, 1), dtype=self.dtype))
