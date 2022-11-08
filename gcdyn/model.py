r"""BDMS inference."""
import jax.numpy as np

# NOTE: sphinx is currently unable to present this in condensed form, using a string type hint
# of "array-like" in the docstring args for now, instead of ArrayLike hint in call signature
# from numpy.typing import ArrayLike

from jax import jit
from jaxopt import ScipyBoundedMinimize
from functools import partial

from gcdyn import responses, mutators
import ete3
from jax.tree_util import register_pytree_node_class


class BdmsModel:
    r"""A class that represents a GC model.

    Args:
        trees: list of trees
        death_rate: Death rate response function.
        mutation_rate: Mutation rate response function.
        mutator: Generator of mutation effects.
        sampling_probability: Probability of sampling a survivor.
        opt_kwargs: keyword arguments to pass to :py:class:`jaxopt.ScipyBoundedMinimize`
    """

    def __init__(
        self,
        trees: list[ete3.TreeNode],
        death_rate: responses.Response,
        mutation_rate: responses.Response,
        mutator: mutators.Mutator,
        sampling_probability: float,
        **opt_kwargs,
    ):
        self.trees = trees
        self.death_rate = death_rate
        self.mutation_rate = mutation_rate
        self.mutator = mutator
        self.sampling_probability = sampling_probability

        self.optimizer = ScipyBoundedMinimize(
            fun=lambda birth_rate: -self.log_likelihood(birth_rate), **opt_kwargs
        )

    # TODO: I think `Responses._param_dict` has no guarantee of the order of the parameters,
    # so I think the init value and optimization bounds might not be specified correctly...
    def fit(
        self,
        init_value: responses.Response = responses.SigmoidResponse(1.0, 0.0, 2.0, 0.0),
        lower_bounds=[0.0, -np.inf, 0.0, 0.0],
        upper_bounds=[np.inf, np.inf, np.inf, np.inf],
    ) -> np.ndarray:
        r"""Given a collection of :py:class:`TreeNode`, fit the parameters of
        the model.

        Args:
            init_value (Response): initial value for the optimizer.
                Result will be of the same functional form.
            lower_bounds (array-like): lower bounds for the optimizer
            upper_bounds (array-like): upper bounds for the optimizer

        All array-like arguments should specify the order of the parameters to match the
        lexographical order of the parameter names (eg. xscale, xshift, yscale, yshift).
        """
        try:
            register_pytree_node_class(type(init_value))
        except ValueError:
            # Already registered this type
            pass

        return self.optimizer.run(
            init_value,
            (np.array(lower_bounds, dtype=float), np.array(upper_bounds, dtype=float)),
        )

    @partial(jit, static_argnums=(0,))
    def log_likelihood(
        self,
        birth_rate: responses.Response,
    ) -> float:
        r"""Compute the log-likelihood of a fully observed tree given the
        specified birth response.

        Args:
            birth_rate: Birth rate response function.
        """

        for tree in self.trees:
            if tree._pruned:
                raise NotImplementedError("tree must be fully observed, not pruned")
            if not tree._sampled:
                raise RuntimeError("tree must be sampled")

        result = 0

        for tree in self.trees:
            for node in tree.iter_descendants():
                Δt = node.dist
                λ = birth_rate(node)
                μ = self.death_rate(node)
                γ = self.mutation_rate(node)
                if not 0 <= self.sampling_probability <= 1:
                    raise ValueError("sampling_probability must be in [0, 1]")
                ρ = self.sampling_probability
                Λ = λ + μ + γ
                logΛ = np.log(Λ)
                # First we have two cases that require special handling of the time interval as part of the
                # likelihood.
                if node.event in (tree._SAMPLING_EVENT, tree._SURVIVAL_EVENT):
                    # exponential survival function (no event before sampling time), then sampling probability
                    result += -Λ * Δt + np.log(
                        ρ if node.event == tree._SAMPLING_EVENT else 1 - ρ
                    )
                elif node.event == tree._MUTATION_EVENT and Δt == 0:
                    # mutation in offspring from birth (simulation run with birth_mutations=True)
                    result += self.mutator.logprob(node.up, node)
                else:
                    # For the rest of the cases, the likelihood is the product of the likelihood of the time
                    # interval (next line, exponential density), then the probability of the given event.
                    result += logΛ - Λ * Δt
                    # multinomial event probability
                    if node.event == tree._BIRTH_EVENT:
                        result += np.log(λ) - logΛ
                    elif node.event == tree._DEATH_EVENT:
                        result += np.log(μ) - logΛ
                    elif node.event == tree._MUTATION_EVENT:
                        result += np.log(γ) - logΛ + self.mutator.logprob(node.up, node)
                    else:
                        raise ValueError(f"unknown event {node.event}")
        return result
