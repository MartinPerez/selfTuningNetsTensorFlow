# Self Tuning Nets (torch port to tensorflow)

This repository was made to port from pytorch to tensorflow the methodology presented in the 2019 ICLR paper "SELF-TUNING NETWORKS: BILEVEL OPTIMIZATION OF HYPERPARAMETERS USING STRUCTURED BEST-RESPONSE FUNCTIONS" was published. Available in https://arxiv.org/pdf/1903.03088.pdf

Few classes and methods are provided to facilitate porting the method to new use cases.
In this repo we show the example of running experiments with:

    * Deep Relu network approximating the X^2 function as in https://arxiv.org/pdf/1903.03088.pdf.
        - Check hyper/experiments/relu_toy_model.py
        - Check notebooks/deep_rely_hyper_trajectories.ipynb

    * Simple convolutional network with added dropout to solve CIFAR 10, similar in spirit to the Keras example in https://www.tensorflow.org/tutorials/images/cnn. Note that this example also demonstrates that it is easy to use a
        - Check hyper/experiments/cnn_models.py
        - Check notebooks/HPO_example.ipynb
        - Check notebooks/CNN_hyper_trajectories.ipynb

## Dev Setup

Setup virtual environment in some location

> python3.6 -m venv self_tuning  
> source self_tuning/bin/activate  

Then install python packages

> cd path/to/self_tuning_nets  
> pip install --upgrade pip setuptools  
> pip install -e .  
> pip install -r tests-requirements.txt  

Launch jupyter server in given port  
Should get the server url after launching it

> export portx=????  
> jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.disable_check_xsrf=True --port=$portx

### Linux version tested

Taken from executing: "cat /etc/os-release"

> NAME="CentOS Linux"  
> VERSION="7 (Core)"  
> ID="centos"  
> ID_LIKE="rhel fedora"  
> VERSION_ID="7"  
> PRETTY_NAME="CentOS Linux 7 (Core)"  
> ANSI_COLOR="0;31"  
> CPE_NAME="cpe:/o:centos:centos:7"  
> HOME_URL="https://www.centos.org/"  
> BUG_REPORT_URL="https://bugs.centos.org/"  
>
> CENTOS_MANTISBT_PROJECT="CentOS-7"  
> CENTOS_MANTISBT_PROJECT_VERSION="7"  
> REDHAT_SUPPORT_PRODUCT="centos"  
> REDHAT_SUPPORT_PRODUCT_VERSION="7"  

### [optional] external software for python animation videos in notebooks

In centOS we need to install some packages for
ffmpeg. Otherwise animation based plotting utilities will not work

> sudo yum -y localinstall --nogpgcheck https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-7.noarch.rpm
>
> sudo yum -y install ffmpeg ffmpeg-devel

### unit testing

Simply go to project folder and use pytest

> cd path/to/self_tuning_test  
> pytest

## Tutorial of experiment setup with hyper library

For complete examples look into:

* hyper/experiments/relu_toy_model.py
* hyper/experiments/cnn_models.py

Here we present a simplified example of how to build an experiment for both data augmentation and a CNN model.

### Definition of hyperparameters

Hyperparameter definitions require providing a unique identifier name string, a function to initialize the unconstrained space tensor in which gradient updates take place, a mapping function from the unconstrained to the constrained space and an initial value for the training perturbations scaling.

    @dataclass
    class HyperParam:
        name: str
        unc_init: tf.python.eager.def_function.Function
        unc_to_con: tf.python.eager.def_function.Function
        scale_init_val: tf.Tensor = tf.constant(0.5, dtype=tf.float64)

Utility function for a bounded continous hyperparameter is provided in hyper/hyperparameters.py, such that is simple to instantiate this common case.

    pixel_keep_prob_param = hyper_param_from_bounds(
        name=f"pixel_keep_prob",
        init_val=0.95,
        min_val=0.25,
        max_val=1.0
    )

Then the Hyperparameters TF module will handle all the variable setup and mappings between the hyperparameters constrained and unconstrained spaces. Such that they can be easily referenced in the training step data function (batch provider) and the network architecture. Using the Hyperparameters class is as simple then as:

    gen = tf.random.Generator.from_seed(FRAMEWORK_SEED)
    hyper_parameters = HyperParameters(
        [pixel_keep_prob_param], BATCH_SIZE, gen
    )

To reference a hyperparameter constrained space batched tensor one simply can

    hyper_parameters.param_con_batch("pixel_keep_prob")

The next step is to provide the weights training and hyperparameters training data functions, where the weights training function can make use of hyperparameters, while the hyperparameters data function, should consist of an independent non augmented dataset that is conceptually used for model selection in the bilevel optimization.

If our input were images on which we drop pixels, for example:

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_images[training_samples], train_labels[training_samples]))
    train_ds = train_ds.shuffle(BATCH_SIZE)\
               .batch(BATCH_SIZE, drop_remainder=True)\
               .repeat()
    train_iter = iter(train_ds)

    dtype=tf.float64
    def weights_data_func():
        X_train, Y_train = train_iter.get_next()
        X_train = tf.cast(X_train, dtype=dtype)
        pixel_drop = hyper_parameters.param_con_batch("pixel_keep_prob")
        new_images = tf.TensorArray(dtype, size=BATCH_SIZE)
        for idx in tf.range(tf.constant(BATCH_SIZE)):
            image = X_train[idx]
            image = image * tf.cast(
                tfd.Bernoulli(probs=pixel_drop[idx][0])\
                .sample(tf.shape(image)), dtype=dtype)
            new_images = new_images.write(idx, image)
        return new_images.stack(), Y_train

Then the hyperparameter training dataset is simply an iterator on other samples

    valid_ds = tf.data.Dataset.from_tensor_slices(
        (train_images[validation_samples], train_labels[validation_samples]))
    valid_ds = valid_ds.shuffle(BATCH_SIZE)\
               .batch(BATCH_SIZE, drop_remainder=True).repeat()
    valid_iter = iter(valid_ds)

    def hyper_data_func():
        X_train, Y_train = valid_iter.get_next()
        return tf.cast(X_train, dtype=dtype), Y_train

In the case of the Network model, like in hyper/nets/custom_cnn.py we should provide the Hyperparameters module during initialization. Here an extract of the module definition:

    class HyperCNN(tf.Module):
    def __init__(self, hyper_params: HyperParameters, name: str = "HyperCNN"):
        super(HyperCNN, self).__init__(name=name)
        self.hyper_params = hyper_params

Because this allows to simply reference hyperparameters values as in __call__ implementation of the module:

    @tf.function
    def __call__(self, x, training=True):
        next_input = x
        for lidx, layer in enumerate(self.conv_layers):
            next_input = layer(next_input, training)
            if lidx in [0, 1]:
                next_input = tf.nn.max_pool2d(next_input, ksize=2, strides=2, padding="VALID")
            if training:
                next_input = dropout2d_layer(next_input, self.hyper_params.param_con_batch(f"dropout{lidx}"))

        next_input = tf.reshape(next_input, (tf.shape(x)[0], self.last_conv_dim))
        next_input = self.lin_layers[0](next_input, training)
        next_input = self.lin_layers[1](next_input, training)

        return next_input

Finally we can take advantage of the weight training and hyperparameter training steps provided in the hyper library to control the training procedure as we desire. The step functions are available in hyper/hyper_training.py:

    weights_training_step, hyperparameters_training_step = get_training_step_functions()

We could for example setup some warmup epochs followed by training epochs:

    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    for step in tqdm(range(warmup_steps),
                     total=warmup_steps, unit="warm_up_batch"):
        weights_training_step(
            model, hyper_parameters, weights_data_func,
            weights_loss_obj, opt_weights, WITH_HYPER_TRAINING
        )

    # training
    steps_per_epoch = 40000 // (BATCH_SIZE * WEIGHT_STEPS)
    training_steps = MAX_EPOCHS * steps_per_epoch
    for step in tqdm(range(training_steps),
                     total=training_steps, unit="training_cycle"):
        for _ in range(WEIGHT_STEPS):
            wloss = weights_training_step(
                model, hyper_parameters, weights_data_func,
                weights_loss_obj, opt_weights, WITH_HYPER_TRAINING
            )
        if WITH_HYPER_TRAINING:
            for _ in range(HYPER_STEPS):
                hloss = hyperparameters_training_step(
                    model, hyper_parameters, hyper_data_func,
                    hyperparam_loss_obj, opt_hyper, opt_scale
                )

Later for metrics we can setup for example an additional testing set to compute an unbiased accuracy to compare with other hyper tuning methods and recover the hyperparameter trajectories, like in:

        if step % steps_per_epoch == 0:
            test_steps = 10000 // BATCH_SIZE
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
            if WITH_HYPER_TRAINING:
                hlosses.append(hloss.numpy())
