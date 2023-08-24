r"""BDMS inference."""

from functools import reduce

import ete3
import jax.numpy as jnp
import numpy as onp
import scipy
import tensorflow as tf
from diffrax import ODETerm, PIDController, Tsit5, diffeqsolve
from jax import lax
import time

# NOTE: sphinx is currently unable to present this in condensed form when the sphinx_autodoc_typehints extension is enabled
from jax.typing import ArrayLike
from jaxopt import ScipyBoundedMinimize

from gcdyn import mutators, poisson

# Pylance can't recognize tf submodules imported the normal way
keras = tf.keras
layers = keras.layers

poisson.set_backend(use_jax=True)


def _select_where(source, selector):
    """`jax.jit`-compatible version of `source[selector]` for a boolean array
    `selector` with exactly one `True` value."""

    return lax.select(selector, source, jnp.zeros_like(source)).sum()


class Callback(tf.keras.callbacks.Callback):
    """Class for control Keras verbosity"""
    epoch = 0
    start_time = time.time()
    last_time = time.time()

    def __init__(self, max_epochs, use_validation=False):
        self.n_between = (5 if max_epochs > 30 else 3) if max_epochs > 10 else 1
        self.use_validation = use_validation

    def on_train_begin(self, epoch, logs=None):
        print('              %s   epoch   total' % ('   valid' if self.use_validation else ''))
        print('   epoch  loss%s    time    time' % ('    loss' if self.use_validation else ''))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_epoch_end(self, batch, logs=None):
        if self.epoch == 0 or self.epoch % self.n_between == 0:
            def lfmt(lv): return ('%7.3f' if lv < 1 else '%7.2f') % lv
            print('  %3d  %s%s    %.1f     %.1f' % (self.epoch, lfmt(logs['loss']), ' ' + lfmt(logs['val_loss']) if self.use_validation else '', time.time() - self.last_time, time.time() - self.start_time))
        self.last_time = time.time()


class BundleMeanLayer(layers.Layer):
    """Assume that the input is of shape (number_of_bundles, bundle_size, feature_dim)."""
    def __init__(self, **kwargs):
        super(BundleMeanLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)  # compute mean over the bundle_size axis

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])  # output shape is (number_of_bundles, feature_dim)


class NeuralNetworkModel:
    """
    Adapts Voznica et. al (2022) for trees with a state attribute.

    Voznica, J., A. Zhukova, V. Boskova, E. Saulnier, F. Lemoine, M. Moslonka-Lefebvre, and O. Gascuel. “Deep Learning from Phylogenies to Uncover the Epidemiological Dynamics of Outbreaks.” Nature Communications 13, no. 1 (July 6, 2022): 3896. https://doi.org/10.1038/s41467-022-31511-0.
    """

    def __init__(
        self,
        encoded_trees: list[onp.ndarray],
        responses: list[list[poisson.Response]],
        bundle_size: int = 50,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.01,
        ema_momentum: float = 0.99,
    ):
        """
        encoded_trees: list of encoded trees
        responses: list of response objects for each tree, i.e. list of lists of responses, with
                   first dimension the same length as encoded_trees, and second dimension with length the
                   number of parameters to predict for each tree. Each response (atm) should just be a constant response
                   with one parameter. (Responses that aren't being estimated need not be provided)
        bundle_size: number of trees to bundle together. The per-bundle mean of predictions is applied to the 
                     convolutional output and then passed to the dense layers.
        """
        leaf_counts = set(
            [len(t[0]) for t in encoded_trees]
        )  # length of first row in encoded tree
        if len(leaf_counts) != 1:
            raise Exception(
                "encoded trees have different lengths: %s"
                % " ".join(str(c) for c in leaf_counts)
            )
        max_leaf_count = list(leaf_counts)[0]
        self.bundle_size = bundle_size
        self.max_leaf_count = max_leaf_count
        self.training_trees = encoded_trees
        self.responses = responses

        self.build_model(dropout_rate, learning_rate, ema_momentum)

    def build_model(self, dropout_rate, learning_rate, ema_momentum, actfn = 'elu'):
        # The TimeDistributed layer wrapper allows us to apply the inner layer (e.g., Conv1D) to each "time step" 
        # of the input tensor independently. In our specific context, our data is structured in the format of:
        # (number_of_examples, bundle_size, feature_dim, max_leaf_count). We're essentially treating the 
        # 'bundle_size' dimension as if it were the "time" or "sequence length" dimension. This means that 
        # for every example, the inner layer is applied to each item within a bundle independently. 

        print('    building model with bundle size %d (%d training trees in %d bundles): dropout %.2f   learn rate %.4f   momentum %.4f' % (self.bundle_size, len(self.training_trees), len(self.training_trees) / self.bundle_size, dropout_rate, learning_rate, ema_momentum))

        num_parameters = sum(len(response._param_dict) for response in self.responses[0])

        network_layers = [
            layers.Lambda(lambda x: tf.transpose(x, (0, 1, 3, 2))),
            layers.TimeDistributed(layers.Conv1D(filters=25, kernel_size=4, activation=actfn)),
            layers.TimeDistributed(layers.Conv1D(filters=25, kernel_size=4, activation=actfn)),
            layers.TimeDistributed(layers.MaxPooling1D(pool_size=2, strides=2)),  # Downsampling by a factor of 2
            layers.TimeDistributed(layers.Conv1D(filters=40, kernel_size=4, activation=actfn)),
            layers.TimeDistributed(layers.GlobalAveragePooling1D()),
            BundleMeanLayer(),
        ]
        dense_unit_list = [48, 32, 16, 8]
        for idense, n_units in enumerate(dense_unit_list):
            network_layers.append(layers.Dense(n_units, activation=actfn))
            if dropout_rate != 0 and idense < len(dense_unit_list) - 1:  # add a dropout layer after each one except the last
                network_layers.append(layers.Dropout(dropout_rate))
        network_layers.append(layers.Dense(num_parameters))

        inputs = keras.Input(shape=(self.bundle_size, 4, self.max_leaf_count))
        outputs = reduce(lambda x, layer: layer(x), network_layers, inputs)
        self.network = keras.Model(inputs=inputs, outputs=outputs)
        self.network.summary(print_fn=lambda x: print("      %s" % x))
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, use_ema=True, ema_momentum=ema_momentum)
        self.network.compile(loss="mean_squared_error", optimizer=optimizer)

    @classmethod
    def _encode_responses(
        cls, responses: list[list[poisson.Response]]
    ) -> onp.ndarray[float]:
        return onp.array(
            [
                onp.hstack([response._flatten()[2] for response in row])  # [2] is the list of parameter values
                for row in responses
            ]
        )

    @classmethod
    def _decode_responses(
        cls,
        response_parameters: list[list[float]],
        example_responses: list[poisson.Response],
    ) -> list[list[poisson.Response]]:
        result = []

        for row in response_parameters:
            responses = []
            i = 0

            response_structure = [response._flatten() for response in example_responses]

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
        return [self._collapse_identical_list(lst) for lst in self._partition_list(list_of_bundles, self.bundle_size)]

    def _reshape_data_wrt_bundle_size(self, data):
        """
        Reshape data to be of shape (num_bundles, bundle_size, 4, max_leaf_count).
        """
        assert data.shape[0] % self.bundle_size == 0
        num_bundles = data.shape[0] // self.bundle_size
        return data.reshape((num_bundles, self.bundle_size, data.shape[1], data.shape[2]))

    def _prepare_trees_for_network_input(self, trees):
        """
        Stack trees and reshape to be of shape (num_bundles, bundle_size, 4, max_leaf_count).
        """
        return self._reshape_data_wrt_bundle_size(onp.stack(trees))

    def fit(self, epochs: int = 30, validation_split: float = 0):
        """Trains neural network on given trees and response parameters."""
        response_parameters = self._encode_responses(self._take_one_identical_item_per_bundle(self.responses))
        self.network.fit(self._prepare_trees_for_network_input(self.training_trees), response_parameters, 
                         epochs=epochs, callbacks=[Callback(epochs, use_validation=validation_split > 0)], verbose=0, validation_split=validation_split
                         ) 

    def predict(
        self,
        encoded_trees: list[onp.ndarray],
    ) -> list[list[poisson.Response]]:
        """Returns the Response objects predicted for each tree."""

        predicted_responses = self.network(self._prepare_trees_for_network_input(encoded_trees))

        return self._decode_responses(
            predicted_responses, example_responses=self.responses[0]
        )


class BirthDeathModel:
    r"""A class that optimizes the rate parameters of a birth-death-mutation-
    sampling model.

    Args:
        log_likelihood: The log likelihood function used to model trees.
        trees: list of trees
        optimized_parameters: Response functions that are passed to `log_likelihood` and that will be
                              optimized via :py:meth:`fit`. Dictionary keys should match the argument
                              names of `log_likelihood`.
        fixed_parameters: Additional response functions and other arguments to `log_likelihood`.
                          Dictionary keys should match the argument names of `log_likelihood`.
        opt_kwargs: Dictionary of keyword arguments to pass to :py:class:`jaxopt.ScipyBoundedMinimize`.
                    The `jit` argument may be of relevance, to JIT-compile the likelihood and its gradient.
    """

    def __init__(
        self,
        log_likelihood: callable,
        trees: list[ete3.TreeNode],
        optimized_parameters: dict[str, poisson.Response],
        fixed_parameters: dict[str, any],
        **opt_kwargs,
    ):
        def objective(optimized_parameters):
            return -log_likelihood(
                trees=trees, **optimized_parameters, **fixed_parameters
            )

        self.parameters = optimized_parameters
        self.objective = objective
        self.optimizer = ScipyBoundedMinimize(fun=objective, **opt_kwargs)

    def fit(
        self,
        lower_bounds: dict[str, ArrayLike] = None,
        upper_bounds: dict[str, ArrayLike] = None,
    ):
        r"""Given a collection of :py:class:`ete3.TreeNode`, fit the parameters
        of the model.

        Args:
            lower_bounds: Dictionary mapping optimized response function names to lower bounds.
            upper_bounds: Dictionary mapping optimized response function names to upper bounds.

        Array values of bounds should specify the order of the parameters to match the
        lexographical order of the parameter names (eg. xscale, xshift, yscale, yshift).
        """

        # Fill in defaults if not provided
        if not lower_bounds:
            lower_bounds = {
                "birth_response": [0.0, -jnp.inf, 0.0, 0.0],
                "death_response": 0,
                "mutation_response": 0,
            }

        if not upper_bounds:
            upper_bounds = {
                "birth_response": [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                "death_response": jnp.inf,
                "mutation_response": jnp.inf,
            }

        # Restrict bounds to optimized parameters
        lower_bounds = {
            param: jnp.array(lower_bounds[param], dtype=float)
            for param in self.parameters
        }

        upper_bounds = {
            param: jnp.array(upper_bounds[param], dtype=float)
            for param in self.parameters
        }

        result = self.optimizer.run(self.parameters, (lower_bounds, upper_bounds))
        self.parameters = result.params

        return result.state

    def log_likelihood(self) -> float:
        r"""Compute the log-likelihood of the current rate parameters given the
        trees."""

        return -self.objective(self.parameters)


def naive_log_likelihood(
    trees: list[ete3.TreeNode],
    birth_response: poisson.Response,
    death_response: poisson.Response,
    mutation_response: poisson.Response,
    mutator: mutators.Mutator,
    extant_sampling_probability: float,
) -> float:
    """
    A model of fully-observed trees, where all survivors and fossils (sampled or unsampled)
    are included in the tree.

    Requires that `tree.prune()` has not been called.
    """
    for tree in trees:
        if tree._pruned:
            raise NotImplementedError("tree must be fully observed, not pruned")
        if not tree._sampled:
            raise RuntimeError("tree must be sampled")

    result = 0

    for tree in trees:
        for node in tree.iter_descendants():
            Δt = node.dist
            if not 0 <= extant_sampling_probability <= 1:
                raise ValueError("sampling_probability must be in [0, 1]")
            ρ = extant_sampling_probability
            parameters = {
                tree._BIRTH_EVENT: birth_response,
                tree._DEATH_EVENT: death_response,
                tree._MUTATION_EVENT: mutation_response,
            }

            # We have two cases that require special handling of the time interval as part of the
            # likelihood.
            if node.event == tree._MUTATION_EVENT and Δt == 0:
                # mutation in offspring from birth (simulation run with birth_mutations=True)
                result += mutator.logprob(node)
            else:
                # waiting time survival function (no event before sampling time), then sampling probability
                result += sum(
                    rate.waiting_time_logsf(node.up, Δt) for rate in parameters.values()
                )
                if node.event in (tree._SAMPLING_EVENT, tree._SURVIVAL_EVENT):
                    result += jnp.log(
                        ρ if node.event == tree._SAMPLING_EVENT else 1 - ρ
                    )
                else:
                    # For the rest of the cases, the likelihood is the product of the likelihood of the time
                    # interval, then the probability of the given event.
                    # Note the log survival function has already been added above.
                    # The next line completes the log pdf of the waiting time for the given event.
                    result += jnp.log(parameters[node.event].λ(node.up, Δt))

                    # For mutations, we need to add the log transition probability
                    if node.event == tree._BIRTH_EVENT:
                        pass
                    elif node.event == tree._DEATH_EVENT:
                        pass
                    elif node.event == tree._MUTATION_EVENT:
                        result += mutator.logprob(node)
                    else:
                        raise ValueError(f"unknown event {node.event}")
    return result


def stadler_appx_log_likelihood(
    trees: list[ete3.TreeNode],
    birth_response: poisson.Response,
    death_response: poisson.Response,
    mutation_response: poisson.Response,
    mutator: mutators.Mutator,
    extant_sampling_probability: float,
    extinct_sampling_probability: float,
    present_time: float,
) -> float:
    """
    A model over trees that are missing unsampled survivors and fossils.
    Assumes that mutations do not occur in unsampled parts of the tree.

    Requires that `tree.prune()` has been called.

    Barido-Sottani, Joëlle, Timothy G Vaughan, and Tanja Stadler. “A Multitype Birth–Death Model for Bayesian Inference of Lineage-Specific Birth and Death Rates.” Edited by Adrian Paterson. Systematic Biology 69, no. 5 (September 1, 2020): 973–86. https://doi.org/10.1093/sysbio/syaa016.
    """
    for tree in trees:
        if not tree._pruned:
            raise NotImplementedError("tree must be pruned")
        if not tree._sampled:
            raise RuntimeError("tree must be sampled")

    result = 0

    for tree in trees:
        for node in tree.iter_descendants():
            Δt = node.dist
            λ = birth_response(node.up)
            μ = death_response(node.up)
            γ = mutation_response(node.up)
            Λ = λ + μ + γ
            ρ = extant_sampling_probability
            σ = extinct_sampling_probability

            if not (0 <= ρ <= 1 and 0 <= σ <= 1):
                raise ValueError("sampling_probability must be in [0, 1]")

            c = jnp.sqrt(Λ**2 - 4 * μ * (1 - σ) * λ)
            x = (-Λ - c) / 2
            y = (-Λ + c) / 2

            def helper(t):
                return (y + λ * (1 - ρ)) * jnp.exp(-c * t) - x - λ * (1 - ρ)

            t_s = present_time - (node.t - Δt)
            t_e = present_time - node.t

            log_f_N = c * (t_e - t_s) + 2 * (
                jnp.log(helper(t_e)) - jnp.log(helper(t_s))
            )

            result += log_f_N

            if node.event == tree._BIRTH_EVENT:
                result += jnp.log(λ)
            elif node.event == tree._DEATH_EVENT:
                result += jnp.log(σ) + jnp.log(μ)
            elif node.event == tree._MUTATION_EVENT:
                result += jnp.log(γ) + mutator.logprob(node)
            elif node.event == tree._SAMPLING_EVENT:
                result += jnp.log(ρ)
            else:
                raise ValueError(f"unknown event {node.event}")

    return result


def stadler_full_log_likelihood(
    trees: list[ete3.TreeNode],
    birth_response: poisson.Response,
    death_response: poisson.Response,
    mutation_response: poisson.Response,
    mutator: mutators.DiscreteMutator,
    extant_sampling_probability: float,
    extinct_sampling_probability: float,
    present_time: float,
    rtol=1e-5,
    atol=1e-9,
    dtmax=0.01,
) -> float:
    """
    A model over trees that are missing unsampled survivors and fossils.

    Requires that `tree.prune()` has been called.
    Requires that a py:class:`mutators.DiscreteMutator` be used as the `mutator`, and that the diagonal of the transition matrix is all zero.
    Currently requires that all py:class:`poisson.Response` objects are homogenous responses.

    Barido-Sottani, Joëlle, Timothy G Vaughan, and Tanja Stadler. “A Multitype Birth–Death Model for Bayesian Inference of Lineage-Specific Birth and Death Rates.” Edited by Adrian Paterson. Systematic Biology 69, no. 5 (September 1, 2020): 973–86. https://doi.org/10.1093/sysbio/syaa016.
    """

    # Ensure our trees are compatible with this model
    for tree in trees:
        if not tree._pruned:
            raise NotImplementedError("tree must be pruned")
        if not tree._sampled:
            raise RuntimeError("tree must be sampled")

    # This likelihood requires a discrete type space to be specified
    type_space = jnp.array(list(mutator.state_space.keys()))
    mutation_probs = mutator.transition_matrix

    # Relevant values to set aside
    λ = birth_response
    μ = death_response
    γ = mutation_response
    ρ = extant_sampling_probability
    σ = extinct_sampling_probability

    if not (0 <= ρ <= 1 and 0 <= σ <= 1):
        raise ValueError("sampling_probability must be in [0, 1]")

    # Compute q along the tree. The likelihood for the tree will be
    # q at the root, times the probabilities of the observed events

    result = 0

    def dp_dt(t, p, args=None):
        # t is a scalar
        # p is a vector matching `type_space`

        return (
            -(
                γ.λ_phenotype(type_space)
                + λ.λ_phenotype(type_space)
                + μ.λ_phenotype(type_space)
            )
            * p
            + μ.λ_phenotype(type_space) * (1 - σ)
            + λ.λ_phenotype(type_space) * p**2
            + γ.λ_phenotype(type_space) * (mutation_probs * p).sum(axis=1)
        )

    def dpq_dt(t, pq, args):
        # Note: if `args` is a jnp.array or np.array for event.up.x, diffrax will trace it for jit
        # and it will have the potential to run quickly (except we can't index into state_space[parent_phenotype])
        # with a tracer object
        # elseif `args` is the TreeNode object directly, diffrax will treat it as a static arg, and this function
        # will recompile every time :( (see equinox filter_jit for details on this automatic dynamic/static detection)
        # Therefore we set args to be event.up.x directly

        # t is a scalar
        # pq is a vector of length `len(type_space)+1`

        p, q_i = pq[:-1], pq[-1]
        parent_phenotype = args

        dq_i = -(
            γ.λ_phenotype(parent_phenotype)
            + λ.λ_phenotype(parent_phenotype)
            + μ.λ_phenotype(parent_phenotype)
        ) * q_i + 2 * λ.λ_phenotype(parent_phenotype) * q_i * _select_where(
            p, type_space == parent_phenotype
        )

        return jnp.hstack([dp_dt(t, p), dq_i])

    for tree in trees:
        for leaf in tree.iter_leaves():
            p = diffeqsolve(
                ODETerm(dp_dt),
                solver=Tsit5(),
                t0=0,
                t1=present_time - leaf.t,
                dt0=0.001,
                y0=jnp.ones_like(type_space) - ρ,
                stepsize_controller=PIDController(rtol=rtol, atol=atol, dtmax=dtmax),
            )

            leaf.p_end = p.ys[-1, :]

        # Postorder over the tree should ensure we integrate from present to past,
        # with initial values computed in correct order & available for every branch
        for event in tree.iter_descendants("postorder"):
            # An event contains the following:
            #  - event.t is the time of the event
            #  - event.dist is the time since the last event
            #  - event.event is the event type
            #  - event.x is the type after this event
            #  - event.up.x is the type that determined the rate parameters that generated this event
            # so the event represents the end of a branch, but contains the type of the next branch

            # Reframe the timing of the branch that leads to this event
            # Note: don't use event.dist here, it introduces more subtractions
            # and we will have floating point issues
            t_start = present_time - event.up.t
            t_end = present_time - event.t

            # We need to get q_i, but only for the type i belonging to the current branch
            if event.event == tree._SAMPLING_EVENT:
                # "a tip at the present t_end == 0"
                event.q_end = jnp.array([ρ])
                # event.p_end already exists
            elif event.event == tree._DEATH_EVENT:
                # "a tip at time t_end > 0"
                event.q_end = jnp.array([μ(event.up) * σ])
                # event.p_end already exists
            elif event.event == tree._BIRTH_EVENT:
                event.q_end = (
                    λ(event.up) * event.children[0].q_start * event.children[1].q_start
                )
                event.p_end = event.children[0].p_start
            elif event.event == tree._MUTATION_EVENT:
                event.q_end = (
                    γ(event.up)
                    * mutator.prob(event.up.x, event.x)
                    * event.children[0].q_start
                )
                event.p_end = event.children[0].p_start
            else:
                raise ValueError(f"unknown event {event.event}")

            pq = diffeqsolve(
                ODETerm(dpq_dt),
                solver=Tsit5(),
                t0=t_end,
                t1=t_start,
                dt0=0.001,
                y0=jnp.hstack([event.p_end, event.q_end]),
                args=event.up.x,
                stepsize_controller=PIDController(rtol=rtol, atol=atol, dtmax=dtmax),
            )

            event.q_start = pq.ys[-1, -1]
            event.p_start = pq.ys[-1, :-1]

        result += jnp.log(tree.children[0].q_start)

    return result


def stadler_full_log_likelihood_scipy(
    trees: list[ete3.TreeNode],
    birth_response: poisson.Response,
    death_response: poisson.Response,
    mutation_response: poisson.Response,
    mutator: mutators.Mutator,
    extant_sampling_probability: float,
    extinct_sampling_probability: float,
    present_time: float,
    **solve_ivp_args,
) -> float:
    """
    A model over trees that are missing unsampled survivors and fossils.
    (JAX-free implementation)

    Requires that `tree.prune()` has been called.
    Requires that a py:class:`mutators.DiscreteMutator` be used as the `mutator`, and that the diagonal of the transition matrix is all zero.
    Currently requires that all py:class:`poisson.Response` objects are homogenous responses.

    Barido-Sottani, Joëlle, Timothy G Vaughan, and Tanja Stadler. “A Multitype Birth–Death Model for Bayesian Inference of Lineage-Specific Birth and Death Rates.” Edited by Adrian Paterson. Systematic Biology 69, no. 5 (September 1, 2020): 973–86. https://doi.org/10.1093/sysbio/syaa016.
    """

    # Ensure our trees are compatible with this model
    for tree in trees:
        if not tree._pruned:
            raise NotImplementedError("tree must be pruned")
        if not tree._sampled:
            raise RuntimeError("tree must be sampled")

    # This likelihood requires a discrete type space to be specified
    assert type(mutator) == mutators.DiscreteMutator
    type_space = onp.array(list(mutator.state_space.keys()))
    assert onp.all(onp.diagonal(mutator.transition_matrix) == 0)
    mutation_probs = mutator.transition_matrix

    # Relevant values to set aside
    λ = birth_response
    μ = death_response
    γ = mutation_response
    ρ = extant_sampling_probability
    σ = extinct_sampling_probability

    if not (0 <= ρ <= 1 and 0 <= σ <= 1):
        raise ValueError("sampling_probability must be in [0, 1]")

    # Compute q along the tree. The likelihood for the tree will be
    # q at the root, times the probabilities of the observed events

    result = 0

    def dp_dt(t, p):
        # t is a scalar
        # p is a vector matching `type_space`

        return (
            -(
                γ.λ_phenotype(type_space)
                + λ.λ_phenotype(type_space)
                + μ.λ_phenotype(type_space)
            )
            * p
            + μ.λ_phenotype(type_space) * (1 - σ)
            + λ.λ_phenotype(type_space) * p**2
            + γ.λ_phenotype(type_space) * (mutation_probs * p).sum(axis=1)
        )

    for tree in trees:
        for leaf in tree.iter_leaves():
            p = scipy.integrate.solve_ivp(
                dp_dt,
                (0, present_time - leaf.t),
                onp.ones_like(type_space) - ρ,
                **solve_ivp_args,
            )

            leaf.p_end = p.y[:, -1]

        def dpq_dt(t, pq):
            p, q_i = pq[:-1], pq[-1]

            dq_i = -(γ(event.up) + λ(event.up) + μ(event.up)) * q_i + 2 * λ(
                event.up
            ) * q_i * onp.squeeze(p[mutator.state_space[event.up.x]])

            return onp.hstack([dp_dt(t, p), dq_i])

        # Postorder over the tree should ensure we integrate from present to past,
        # with initial values computed in correct order & available for every branch
        for event in tree.iter_descendants("postorder"):
            # An event contains the following:
            #  - event.t is the time of the event
            #  - event.dist is the time since the last event
            #  - event.event is the event type
            #  - event.x is the type after this event
            #  - event.up.x is the type that determined the rate parameters that generated this event
            # so the event represents the end of a branch, but contains the type of the next branch

            # Reframe the timing of the branch that leads to this event
            # Note: don't use event.dist here, it introduces more subtractions
            # and we will have floating point issues
            t_start = present_time - event.up.t
            t_end = present_time - event.t

            # We need to get q_i, but only for the type i belonging to the current branch
            if event.event == tree._SAMPLING_EVENT:
                # "a tip at the present t_end == 0"
                assert t_end == 0
                event.q_end = onp.array([ρ])
                # event.p_end already exists
            elif event.event == tree._DEATH_EVENT:
                # "a tip at time t_end > 0"
                event.q_end = onp.array([μ(event.up) * σ])
                # event.p_end already exists
            elif event.event == tree._BIRTH_EVENT:
                event.q_end = (
                    λ(event.up) * event.children[0].q_start * event.children[1].q_start
                )
                event.p_end = event.children[0].p_start
            elif event.event == tree._MUTATION_EVENT:
                event.q_end = (
                    γ(event.up)
                    * mutator.prob(event.up.x, event.x)
                    * event.children[0].q_start
                )
                event.p_end = event.children[0].p_start
            else:
                raise ValueError(f"unknown event {event.event}")

            q_grid = scipy.integrate.solve_ivp(
                dpq_dt,
                (t_end, t_start),
                onp.hstack([event.p_end, event.q_end]),
                **solve_ivp_args,
            )

            event.q_start = q_grid.y[-1, -1]
            event.p_start = q_grid.y[:-1, -1]

        result += onp.log(tree.children[0].q_start)

    return result
