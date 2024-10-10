r"""Neural network models for inference."""
# fmt: off

import numpy as onp
import tensorflow as tf
import time
import sys
from functools import reduce
import math

from gcdyn import poisson, encode, utils

poisson.set_backend(use_jax=True)

# Pylance can't recognize tf submodules imported the normal way
import keras  # these imports are really slow
layers = keras.layers


class Callback(tf.keras.callbacks.Callback):
    """Class to control Keras verbosity"""
    epoch = 0
    start_time = time.time()
    last_time = time.time()

    def __init__(self, max_epochs, use_validation=False):
        self.n_between = (5 if max_epochs > 30 else 3) if max_epochs > 10 else 1
        self.use_validation = use_validation

    def on_train_begin(self, epoch, logs=None):
        print("              %s   epoch   total" % ("   valid" if self.use_validation else ""))
        print("   epoch  loss%s    time    time" % ("    loss" if self.use_validation else ""))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_epoch_end(self, batch, logs=None):
        if self.epoch == 0 or self.epoch % self.n_between == 0:
            print("  {:3d}  {:7.3f}{:7.3f}    {:.1f}     {:.1f}".format(self.epoch, logs["loss"], logs["val_loss"], time.time() - self.last_time, time.time() - self.start_time))
            sys.stdout.flush()
        self.last_time = time.time()

# @keras.saving.register_keras_serializable()
class BundleMeanLayer(layers.Layer):
    """Assume that the input is of shape (number_of_bundles, bundle_size, feature_dim)."""

    def __init__(self, **kwargs):
        super(BundleMeanLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)  # compute mean over the bundle_size axis

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[2],
        )  # output shape is (number_of_bundles, feature_dim)

# @keras.saving.register_keras_serializable()
class SigmoidTransposeLayer(layers.Layer):
    """Transpose inputs (used to be Lambda layer, but they're not really serializable."""

    def __init__(self, **kwargs):
        super(SigmoidTransposeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # leave first (batch/sample) and second (bundle) dimensions the same, and swap third (dist/phenotyp) and fourth (leaf/internal node) dimensions
        return tf.transpose(inputs, perm=(0, 1, 3, 2))

# @keras.saving.register_keras_serializable()
class PerCellTransposeLayer(layers.Layer):
    """Transpose inputs (used to be Lambda layer, but they're not really serializable."""

    def __init__(self, **kwargs):
        super(PerCellTransposeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # leave first (batch/sample) and second (bundle) dimensions the same, and swap third (dist/phenotyp) and fourth (leaf/internal node) dimensions
        return tf.transpose(inputs, perm=(0, 2, 1))

# ----------------------------------------------------------------------------------------
# @keras.saving.register_keras_serializable(package='gcdyn_nn', name='curve_loss_fcn')
@tf.function  # NOTE need to comment this decorator in order for .numpy() to work (and add run_eagerly to .compile() call, but note that training is WAY the fuck slower if it's commented
def curve_loss(y_true, y_pred):  # copied from/modeled after utils.resp_fcn_diff()
    # ----------------------------------------------------------------------------------------
    def params(yt):
        assert utils.sigmoid_params == ['xscale', 'xshift', 'yscale']  # ick
        pvlist = tf.split(tf.cast(yt, tf.float64), [1, 1, 1], axis=1)  # axis 0 is N samples
        return {p : v for p, v in zip(utils.sigmoid_params, pvlist)}
    # ----------------------------------------------------------------------------------------
    def sigval(xv, pvals):
        return pvals['yscale'] * tf.math.sigmoid(pvals['xscale'] * (xv - pvals['xshift']))
    # ----------------------------------------------------------------------------------------
    def print_debug():  # NOTE to run this, you probably need to comment @tf.function above, as well as add run_eagerly=True in .compile()
        print('params:', y_true, y_pred)
        print('     true    ', true_svals.numpy())
        print('     pred    ', pred_svals.numpy())
        print('     diff    ', (true_svals - pred_svals).numpy())
        print('     abs diff', tf.math.abs(true_svals - pred_svals).numpy())
        print('    sum %.3f   diff %.5f' % (tf.reduce_sum(tf.math.abs(true_svals - pred_svals) * tf.constant(dx)).numpy(), normed_area.numpy()))
    # ----------------------------------------------------------------------------------------
    xbounds = [-5, 5]  # once we're passing in input affinity values, we won't need this somewhat abritrary choice
    nsteps = 1000  # having this large (like 1k) seems sometimes important for training stability (and weirdly doesn't seem to slow down training)
    dx = (xbounds[1] - xbounds[0]) / nsteps
    xvals = tf.constant(list(onp.arange(xbounds[0], 0, dx)) + list(onp.arange(0, xbounds[1] + dx, dx)), dtype=tf.float64)
    true_svals, pred_svals = [sigval(xvals, params(yv)) for yv in [y_true, y_pred]]
    sumv = tf.reduce_sum(tf.math.abs(true_svals - pred_svals) * tf.constant(dx, dtype=tf.float64))
    all_yvals = tf.concat([true_svals, pred_svals], 0)
    xdist = tf.math.abs(tf.constant(xbounds[1] - xbounds[0], dtype=tf.float64))
    ydist = tf.math.abs(tf.math.reduce_max(all_yvals) - tf.math.reduce_min(all_yvals))
    rect_area = tf.reduce_sum(xdist * ydist)  # area of whole plot/rectangle
    normed_area = sumv / rect_area
    # print_debug()
    return normed_area

# ----------------------------------------------------------------------------------------
# @keras.saving.register_keras_serializable()
# This seems to be really important for training stability (without it, it often gets stuck at very large parameter values)
# It might also be nice to move this to a command line arg
def clipfcn(x, minv=[0.001, -1.5, 0.1], maxv=[3.5, 5, 65]):  # NOTE these should be significantly wider than the simulation bounds
    return tf.clip_by_value(x, clip_value_min=minv, clip_value_max=maxv)

# ----------------------------------------------------------------------------------------
class ParamNetworkModel:
    """
    Adapts Voznica et. al (2022) for trees with a state attribute.
    Voznica, J., A. Zhukova, V. Boskova, E. Saulnier, F. Lemoine, M. Moslonka-Lefebvre, and O. Gascuel. “Deep Learning from Phylogenies to Uncover the Epidemiological Dynamics of Outbreaks.” Nature Communications 13, no. 1 (July 6, 2022): 3896. https://doi.org/10.1038/s41467-022-31511-0.
    Predicts the parameters of the response functions (atm, this means the three parameters of a sigmoid birth response, although minimal/no changes here would be required from some generalizations).
    """

    def __init__(self, example_response_list, bundle_size=50, custom_loop=False):
        """
        example_response_list: list of response objects for a single tree (used for two things: to work out the number of
                     parameters, and how to encode/decode by calling its _flatten() fcn)
        bundle_size: number of trees to bundle together. The per-bundle mean of predictions is applied to the
                     convolutional output and then passed to the dense layers.
        """
        self.example_response_list = example_response_list
        self.num_parameters = sum(
            len(rsp._param_dict) for rsp in self.example_response_list
        )
        self.bundle_size = bundle_size
        self.custom_loop = custom_loop
        self.no_class_model = False  # TODO should remove this eventually, once I decide serialization is final

    def build_model(
        self,
        max_leaf_count: float,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.01,
        ema_momentum: float = 0.99,
        prebundle_layer_cfg: str = 'default',
        loss_fcn="mean_squared_error",
        actfn: str = "elu",
    ):
        # The TimeDistributed layer wrapper allows us to apply the inner layer (e.g., Conv1D) to each "time step"
        # of the input tensor independently. In our specific context, our data is structured in the format of:
        # (number_of_examples, bundle_size, feature_dim, max_leaf_count). We're essentially treating the
        # 'bundle_size' dimension as if it were the "time" or "sequence length" dimension. This means that
        # for every example, the inner layer is applied to each item within a bundle independently.

        self.max_leaf_count = max_leaf_count
        self.actfn = actfn
        print("    building model with bundle size %d: dropout %.2f   learn rate %.4f   momentum %.4f" % (self.bundle_size, dropout_rate, learning_rate, ema_momentum))

        # various config options for the layers before the bundle mean layer
        self.pre_bundle_layers = {
            'small' : [
                layers.Conv1D(filters=25, kernel_size=4, activation=actfn),
                layers.GlobalAveragePooling1D(),  # one number for each filter from the previous layer
            ],
            'default' : [
                layers.Conv1D(filters=25, kernel_size=4, activation=actfn),
                layers.Conv1D(filters=25, kernel_size=4, activation=actfn),
                layers.MaxPooling1D(pool_size=2, strides=2),  # Downsampling by a factor of 2
                layers.Conv1D(filters=40, kernel_size=4, activation=actfn),
                layers.GlobalAveragePooling1D(),  # one number for each filter from the previous layer
            ],
            'big' : [
                layers.Conv1D(filters=50, kernel_size=4, activation=actfn),
                layers.Conv1D(filters=50, kernel_size=4, activation=actfn),
                layers.Conv1D(filters=50, kernel_size=4, activation=actfn),
                layers.MaxPooling1D(pool_size=2, strides=2),  # Downsampling by a factor of 2
                layers.Conv1D(filters=60, kernel_size=4, activation=actfn),
                layers.Conv1D(filters=60, kernel_size=4, activation=actfn),
                layers.GlobalAveragePooling1D(),  # one number for each filter from the previous layer
            ],
        }

        # TODO remove this once you've settled things down a bit more
        if self.no_class_model:  # trivial model for testing serialization without our custom layers (didn't actually get the serialization working since i found the compile=False arg to load_model())
            network_layers = []
            for tlr in self.pre_bundle_layers[prebundle_layer_cfg]:
                network_layers.append(tlr)
            network_layers.append(layers.Dense(self.num_parameters, activation=clipfcn))  # clipfcn is important for training stability
            inputs = keras.Input(shape=(self.max_leaf_count, 4))
        else:
            network_layers = [SigmoidTransposeLayer()]
            for tlr in self.pre_bundle_layers[prebundle_layer_cfg]:
                network_layers.append(layers.TimeDistributed(tlr))
            network_layers.append(BundleMeanLayer())  # combine predictions from all trees in bundle
            dense_unit_list = [48, 32, 16, 8]
            for idense, n_units in enumerate(dense_unit_list):
                network_layers.append(layers.Dense(n_units, activation=self.actfn))
                if dropout_rate != 0 and idense < len(dense_unit_list) - 1:  # add a dropout layer after each one except the last
                    network_layers.append(layers.Dropout(dropout_rate))
            network_layers.append(layers.Dense(self.num_parameters, activation=clipfcn))  # clipfcn is important for training stability
            inputs = keras.Input(shape=(self.bundle_size, 4, self.max_leaf_count))

        outputs = reduce(lambda x, layer: layer(x), network_layers, inputs)
        self.network = keras.Model(inputs=inputs, outputs=outputs)
        # def pfn(x, line_break=False): print("      %s" % x)  # this doesn't seem to work with keras 3
        self.network.summary() #print_fn=pfn)
        self.optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, use_ema=True, ema_momentum=ema_momentum
        )
        self.loss_fcn = curve_loss if loss_fcn=='curve' else loss_fcn
        if not self.custom_loop:
            self.network.compile(loss=curve_loss if loss_fcn=='curve' else self.loss_fcn, optimizer=self.optimizer)  # turn on this to allow to call .numpy() on tf tensors to get float value: , run_eagerly=True)

    @classmethod
    def _encode_responses(
        cls, responses: list[list[poisson.Response]]
    ) -> onp.ndarray[float]:
        return onp.array(
            [
                onp.hstack(
                    [response._flatten()[2] for response in row]
                )  # [2] is the list of parameter values
                for row in responses
            ]
        )

    @classmethod
    def _decode_responses(
        cls,
        response_parameters: list[list[float]],
        example_response_list: list[poisson.Response],
    ) -> list[list[poisson.Response]]:
        result = []

        for row in response_parameters:
            responses = []
            i = 0

            response_structure = [response._flatten() for response in example_response_list]

            for (
                response_cls,
                param_names,
                example_param_values,
            ) in response_structure:
                param_values = row[i : i + len(example_param_values)]
                responses.append(response_cls._from_flat(param_names, param_values))
                i += len(example_param_values)

            result.append(responses)

        return result

    @staticmethod
    def _partition_list(input_list, sublist_len):
        """
        Partitions a given list into sublists of size sublist_len.
        """
        if len(input_list) % sublist_len != 0:
            raise ValueError(
                f"The length of the input list ({len(input_list)}) is not divisible by {sublist_len}"
            )
        return [
            input_list[i : i + sublist_len]
            for i in range(0, len(input_list), sublist_len)
        ]

    @staticmethod
    def _collapse_identical_list(lst):
        if not lst:
            raise ValueError("List is empty")

        first_element = lst[0]

        for item in lst:
            if item != first_element:
                raise ValueError(
                    f"All items in the list are not identical: {first_element} vs {item}"
                )

        return first_element

    def _take_one_identical_item_per_bundle(self, list_of_bundles):
        """
        list_of_bundles is a flat list with bundles of identical items, e.g. [3,3,5,5,8,8].
        """
        return [
            self._collapse_identical_list(lst)
            for lst in self._partition_list(list_of_bundles, self.bundle_size)
        ]

    def _reshape_data_wrt_bundle_size(self, data):
        """
        Reshape data to be of shape (num_bundles, bundle_size, 4, max_leaf_count).
        """
        if self.no_class_model:
            return data.reshape(
                (num_bundles, data.shape[2], data.shape[1])
            )
        else:
            assert data.shape[0] % self.bundle_size == 0
            num_bundles = data.shape[0] // self.bundle_size
            return data.reshape(
                (num_bundles, self.bundle_size, data.shape[1], data.shape[2])
            )

    def _prepare_trees_for_network_input(self, trees):
        """
        Stack trees and reshape to be of shape (num_bundles, bundle_size, 4, max_leaf_count).
        """
        return self._reshape_data_wrt_bundle_size(onp.stack(trees))

    def fit(self, training_trees: list[onp.ndarray], responses: list[list[poisson.Response]], epochs: int = 30, batch_size: int = None, validation_split: float = 0):
        """
        Trains neural network on given trees and responses.
        training_trees: list of encoded trees
        responses: list of response objects for each tree, i.e. list of lists of responses, with
                   first dimension the same length as training_trees, and second dimension with length the
                   number of parameters to predict for each tree. Each response (atm) should just be a constant response
                   with one parameter. (Responses that aren't being estimated need not be provided)
        """
        # ----------------------------------------------------------------------------------------
        @tf.function  # NOTE need to comment this decorator in order for .numpy() to work (and add run_eagerly to .compile() call, but note that training is WAY slower if it's commented
        def train_fcn(xbtch, ybtch):
            with tf.GradientTape() as tape:  # operations are recorded on the tape
                logits = self.network(xbtch, training=True)
                btch_loss = self.loss_fcn(ybtch, logits)
            grads = tape.gradient(btch_loss, self.network.trainable_weights)  # retrieve gradients
            self.optimizer.apply(grads, self.network.trainable_weights)  # run one gradient descent step
            return btch_loss
        # ----------------------------------------------------------------------------------------
        print("    fitting model with bundle size %d (%d training trees in %d bundles)" % (self.bundle_size, len(training_trees), len(training_trees) / self.bundle_size))
        response_parameters = self._encode_responses(self._take_one_identical_item_per_bundle(responses))

        if self.custom_loop:  # this more or less works, but gives different loss value (shouldn't it be identical ish?), seems less stable, and seems also quite a bit slower
            print('    using custom training loop')
            x_values, y_values = self._prepare_trees_for_network_input(training_trees), response_parameters
            assert len(x_values) == len(y_values)
            n_train = int(len(x_values) * (1 - validation_split))
            x_val = x_values[n_train:]
            y_val = y_values[n_train:]
            x_train = x_values[:n_train]
            y_train = y_values[:n_train]

            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            val_dataset = val_dataset.batch(batch_size)

            callbacks = tf.keras.callbacks.CallbackList([Callback(epochs, use_validation=validation_split > 0)])
            callbacks.on_train_begin()
            for epoch in range(epochs):
                callbacks.on_epoch_begin(epoch)
                for x_batch_train, y_batch_train in train_dataset:
                    btch_loss = train_fcn(x_batch_train, y_batch_train)
                    val_loss = tf.constant(0, dtype=tf.float64)
                    for x_batch_val, y_batch_val in val_dataset:  # i don't think there's any actual reason to use batches on validation, but not sure how else to arrange it
                        val_logits = self.network(x_batch_val, training=False)
                        val_loss += self.loss_fcn(y_batch_val, val_logits)
                logs = {'loss' : btch_loss, 'val_loss' : val_loss}
                callbacks.on_epoch_end(epoch, logs=logs)
        else:
            print('    using network.fit')
            self.network.fit(
                self._prepare_trees_for_network_input(training_trees),
                response_parameters,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[Callback(epochs, use_validation=validation_split > 0)],
                verbose=0,
                validation_split=validation_split,
                # validation_data=val_dataset,
            )

    def predict(
        self,
        encoded_trees: list[onp.ndarray],
    ) -> list[list[poisson.Response]]:
        """Returns the Response objects predicted for each tree."""

        predicted_responses = self.network(
            self._prepare_trees_for_network_input(encoded_trees)
        )

        return self._decode_responses(predicted_responses, example_response_list=self.example_response_list)

    def load(self, fname):
        print('    reading model file from %s' % fname)
        with keras.utils.custom_object_scope({'BundleMeanLayer': BundleMeanLayer, 'SigmoidTransposeLayer' : SigmoidTransposeLayer, 'curve_loss' : curve_loss, 'clipfcn' : clipfcn}):
            # NOTE that compile=False is fine since we're only calling .predict(), but if we wanted to e.g. continue training from the loaded model, we'd need to compile it (and getting serialization working [with compilation] was a huge clusterfuck)
            self.network = keras.models.load_model(fname, safe_mode=False, compile=False)

# ----------------------------------------------------------------------------------------
@tf.function
# I think this works, but I haven't tested a ton since the per-cell prediction isn't really working well evefn with constant N leaves
def per_cell_masked_loss(y_true, y_pred):
    loss = tf.square(y_true - y_pred)
    loss = tf.where(tf.cast(encode.is_empty(y_true), tf.bool), tf.zeros_like(loss), loss)  # if filled (fill_true == 1) use <loss>, otherwise set to zero (NOTE fill_pred is [probably?] useless)
    return tf.reduce_sum(loss)  # tf.reduce_mean(loss)

# ----------------------------------------------------------------------------------------
class PerCellNetworkModel:
    """
    Similar to ParamNetworkModel, except we directly predict the fitness of each cell (rather than the parameters of the fitenss response function).
    """

    def __init__(self):
        pass

    # ----------------------------------------------------------------------------------------
    def build_model(
        self,
        max_leaf_count: float,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.01,
        ema_momentum: float = 0.99,
        loss_fcn="mean_squared_error",
        actfn: str = "elu",
    ):
        self.max_leaf_count = max_leaf_count
        self.actfn = actfn
        print("    building model with: dropout %.2f   learn rate %.4f   momentum %.4f" % (dropout_rate, learning_rate, ema_momentum))

        tlayers = [
            layers.Conv1D(filters=25, kernel_size=4, activation=actfn),
            layers.Conv1D(filters=25, kernel_size=4, activation=actfn),
            layers.MaxPooling1D(pool_size=2, strides=2),  # Downsampling by a factor of 2
            layers.Conv1D(filters=40, kernel_size=4, activation=actfn),
            layers.GlobalAveragePooling1D(),  # one number for each filter from the previous layer
        ]

        network_layers = [PerCellTransposeLayer()]
        for tlr in tlayers:
            network_layers.append(tlr)
        dense_unit_list = [48, 32, 16, 8]
        for idense, n_units in enumerate(dense_unit_list):
            network_layers.append(layers.Dense(n_units, activation=self.actfn))
            if dropout_rate != 0 and idense < len(dense_unit_list) - 1:  # add a dropout layer after each one except the last
                network_layers.append(layers.Dropout(dropout_rate))
        network_layers.append(layers.Flatten())
        network_layers.append(layers.Dense(encode.mtx_lens['fitness'] * self.max_leaf_count))
        network_layers.append(layers.Reshape((encode.mtx_lens['fitness'], self.max_leaf_count)))

        inputs = keras.Input(shape=(encode.mtx_lens['tree'], self.max_leaf_count))
        outputs = reduce(lambda x, layer: layer(x), network_layers, inputs)
        self.network = keras.Model(inputs=inputs, outputs=outputs)
        # def pfn(x, line_break=False): print("      %s" % x)  # this doesn't seem to work with keras 3
        self.network.summary() #print_fn=pfn)
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, use_ema=True, ema_momentum=ema_momentum
        )
        self.network.compile(loss=loss_fcn, optimizer=optimizer) #, run_eagerly=True)  # turn on this to allow to call .numpy() on tf tensors to get float value: , run_eagerly=True)

    # ----------------------------------------------------------------------------------------
    def fit(self, training_trees, training_fitnesses, epochs=30, batch_size=None, validation_split=0):
        self.network.fit(
            onp.stack(training_trees),
            onp.stack(training_fitnesses),
            epochs=epochs,
            callbacks=[Callback(epochs, use_validation=validation_split > 0)],
            verbose=0,
            validation_split=validation_split,
        )

    # ----------------------------------------------------------------------------------------
    def predict(self, encoded_trees):
        predicted_fitnesses = self.network(onp.stack(encoded_trees))
        return predicted_fitnesses

    # ----------------------------------------------------------------------------------------
    def load(self, fname):
        # NOTE see other load fcn above if this isn't working
        print('    reading model file from %s' % fname)
        self.network = keras.models.load_model(fname, safe_mode=False)

# fmt: on
