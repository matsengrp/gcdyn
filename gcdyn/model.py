r"""BDMS inference
------------------

"""

from __future__ import annotations

# NOTE: sphinx is currently unable to present this in condensed form when the sphinx_autodoc_typehints extension is enabled
from jax.typing import ArrayLike

import jax.numpy as np

from jax import jit
from jaxopt import ScipyBoundedMinimize
from functools import partial

from gcdyn import poisson, mutators
import ete3


class BDMSModel:
    r"""A class that represents a BDMS model.

    Args:
        trees: list of trees
        birth_rate: Initial birth rate response function. If `birth_rate.grad == True`, will be optimizable via :py:meth:`fit`.
        death_rate: Initial death rate response function. If `death_rate.grad == True`, will be optimizable via :py:meth:`fit`.
        mutation_rate: Initial mutation rate response function. If `mutation_rate.grad == True`, will be optimizable via :py:meth:`fit`.
        mutator: Generator of mutation effects.
        sampling_probability: Probability of sampling a survivor.
        opt_kwargs: keyword arguments to pass to :py:class:`jaxopt.ScipyBoundedMinimize`
    """

    def __init__(
        self,
        trees: list[ete3.TreeNode],
        birth_rate: poisson.Response,
        death_rate: poisson.Response,
        mutation_rate: poisson.Response,
        mutator: mutators.Mutator,
        sampling_probability: float,
        **opt_kwargs,
    ):
        self.initialize_parameters(birth_rate, death_rate, mutation_rate)
        for event, rate in self.parameters.items():
            if not isinstance(rate, poisson.HomogeneousResponse):
                raise NotImplementedError(
                    f"Non-homogeneous {event} rate not implemented"
                )

        self._trees = trees
        self._mutator = mutator
        self._sampling_probability = sampling_probability

        @jit
        def objective(optimizable_params, fixed_params):
            return self._neg_log_likelihood({**optimizable_params, **fixed_params})

        self.optimizer = ScipyBoundedMinimize(fun=objective, **opt_kwargs)

    def fit(
        self,
        birth_rate_lower_bound: ArrayLike = [0.0, -np.inf, 0.0, 0.0],
        birth_rate_upper_bound: ArrayLike = [np.inf, np.inf, np.inf, np.inf],
        death_rate_lower_bound: ArrayLike = 0,
        death_rate_upper_bound: ArrayLike = np.inf,
        mutation_rate_lower_bound: ArrayLike = 0,
        mutation_rate_upper_bound: ArrayLike = np.inf,
    ):
        r"""Given a collection of :py:class:`ete3.TreeNode`, fit the parameters
        of the model.

        Args:
            birth_rate_lower_bound: lower bounds for the birth rate parameters
            birth_rate_upper_bound: upper bounds for the birth rate parameters
            death_rate_lower_bound: lower bounds for the death rate parameters
            death_rate_upper_bound: upper bounds for the death rate parameters
            mutation_rate_lower_bound: lower bounds for the mutation rate parameters
            mutation_rate_upper_bound: upper bounds for the mutation rate parameters

        Array arguments should specify the order of the parameters to match the
        lexographical order of the parameter names (eg. xscale, xshift, yscale, yshift).
        """

        init_value = {name: val for name, val in self.parameters.items() if val.grad}
        fixed_params = {
            name: val for name, val in self.parameters.items() if not val.grad
        }

        lower_bounds = {
            "birth": np.asarray(birth_rate_lower_bound, dtype=float),
            "death": np.asarray(death_rate_lower_bound, dtype=float),
            "mutation": np.asarray(mutation_rate_lower_bound, dtype=float),
        }

        lower_bounds = {
            name: val for name, val in lower_bounds.items() if name in init_value
        }

        upper_bounds = {
            "birth": np.asarray(birth_rate_upper_bound, dtype=float),
            "death": np.asarray(death_rate_upper_bound, dtype=float),
            "mutation": np.asarray(mutation_rate_upper_bound, dtype=float),
        }

        upper_bounds = {
            name: val for name, val in upper_bounds.items() if name in init_value
        }

        for v in init_value.values():
            poisson._register_with_pytree(type(v))

        result = self.optimizer.run(
            init_value, (lower_bounds, upper_bounds), fixed_params
        )

        self.initialize_parameters(**result.params, **fixed_params)

        return result.state

    def log_likelihood(self) -> float:
        r"""Compute the log-likelihood of the birth parameters given fully
        observed trees."""

        return -self._neg_log_likelihood(self.parameters)

    def initialize_parameters(
        self,
        birth_rate: poisson.Response,
        death_rate: poisson.Response,
        mutation_rate: poisson.Response,
    ) -> None:
        r"""Updates the current values of the model parameters."""

        self.parameters = {
            "birth": birth_rate,
            "death": death_rate,
            "mutation": mutation_rate,
        }

        for v in self.parameters.values():
            poisson._register_with_pytree(type(v))

    @partial(jit, static_argnums=(0,))
    def _neg_log_likelihood(
        self,
        parameters: dict[str, poisson.Response],
    ) -> float:
        for tree in self._trees:
            if tree._pruned:
                raise NotImplementedError("tree must be fully observed, not pruned")
            if not tree._sampled:
                raise RuntimeError("tree must be sampled")

        result = 0

        for tree in self._trees:
            for node in tree.iter_descendants():
                Δt = node.dist
                if not 0 <= self._sampling_probability <= 1:
                    raise ValueError("sampling_probability must be in [0, 1]")
                ρ = self._sampling_probability

                # We have two cases that require special handling of the time interval as part of the
                # likelihood.
                if node.event == tree._MUTATION_EVENT and Δt == 0:
                    # mutation in offspring from birth (simulation run with birth_mutations=True)
                    result += self._mutator.logprob(node)
                else:
                    # waiting time survival function (no event before sampling time), then sampling probability
                    result += sum(
                        rate.waiting_time_logsf(node.up, Δt)
                        for rate in parameters.values()
                    )
                    if node.event in (tree._SAMPLING_EVENT, tree._SURVIVAL_EVENT):
                        result += np.log(
                            ρ if node.event == tree._SAMPLING_EVENT else 1 - ρ
                        )
                    else:
                        # For the rest of the cases, the likelihood is the product of the likelihood of the time
                        # interval, then the probability of the given event.
                        # Note the log survival function has already been added above.
                        # The next line completes the log pdf of the waiting time for the given event.
                        result += np.log(parameters[node.event].λ(node.up, Δt))

                        # For mutations, we need to add the log transition probability
                        if node.event == tree._BIRTH_EVENT:
                            pass
                        elif node.event == tree._DEATH_EVENT:
                            pass
                        elif node.event == tree._MUTATION_EVENT:
                            result += self._mutator.logprob(node)
                        else:
                            raise ValueError(f"unknown event {node.event}")
        return -result
