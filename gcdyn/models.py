r"""BDMS inference."""

import ete3
import jax.numpy as jnp
import numpy as onp
import scipy
from diffrax import ODETerm, PIDController, Tsit5, diffeqsolve
from jax import lax

# NOTE: sphinx is currently unable to present this in condensed form when the sphinx_autodoc_typehints extension is enabled
from jax.typing import ArrayLike
from jaxopt import ScipyBoundedMinimize

from gcdyn import mutators, poisson

poisson.set_backend(use_jax=True)


def _select_where(source, selector):
    """`jax.jit`-compatible version of `source[selector]` for a boolean array
    `selector` with exactly one `True` value."""

    return lax.select(selector, source, jnp.zeros_like(source)).sum()


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
    assert isinstance(mutator, mutators.DiscreteMutator)  # NOTE lint is forcing me (duncan) to use isinstance instead of comparing types, but I don't know anything about this fcn and I don't think it runs in tests
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
