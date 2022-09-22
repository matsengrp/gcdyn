r"""BDMS inference.

.. todo::

    Harmonize/integrate with :py:mod:`gcdyn.bdms` module.
"""

import jax.numpy as np
from numpy.typing import ArrayLike
from jax.scipy.stats import norm
from jax import jit
from jax.scipy.special import expit
from jaxopt import ScipyBoundedMinimize
from functools import partial

from gcdyn.bdms import TreeNode


class Model:
    r"""A class that represents a GC model.

    Args:
        trees: list of trees
        μ: death rate
        γ: mutation rate
        ρ: sampling probability
        opt_kwargs: keyword arguments to pass to :py:class:`jaxopt.ScipyBoundedMinimize`
    """

    def __init__(
        self, trees: list[TreeNode], μ: float, γ: float, ρ: float, **opt_kwargs
    ):
        self.trees = trees
        self.μ = μ
        self.γ = γ
        self.ρ = ρ

        self.optimizer = ScipyBoundedMinimize(
            fun=jit(lambda θ: -self.log_likelihood(θ)), **opt_kwargs
        )

    def λ(self, x: float, θ) -> float:
        r"""Birth rate of phenotype x.

        Args:
            x: phenotype

        Returns:
            float: birth rate
        """
        return θ[0] * expit(θ[1] * (x - θ[2])) + θ[3]

    def fit(
        self,
        init_value: ArrayLike = [2.0, 1.0, 0.0, 0.0],
        lower_bounds: ArrayLike = [0.0, 0.0, -np.inf, 0.0],
        upper_bounds: ArrayLike = [np.inf, np.inf, np.inf, np.inf],
    ) -> ArrayLike:
        r"""Given a collection of :py:class:`TreeNode`, fit the parameters of
        the model.

        Args:
            init_value: initial value for the optimizer
            lower_bounds: lower bounds for the optimizer
            upper_bounds: upper bounds for the optimizer
        """
        return self.optimizer.run(
            np.array(init_value), (np.array(lower_bounds), np.array(upper_bounds))
        )

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, θ) -> float:
        r"""Compute log likelihood of parameters.

        Args:
            θ: parameters
        """
        result = 0
        for tree in self.trees:
            for node in tree.children[0].traverse():
                x = node.up.x
                Δt = node.dist
                λ_x = self.λ(x, θ)
                Λ = λ_x + self.μ + self.γ
                logΛ = np.log(Λ)
                if node.event in ("sampling", "survival"):
                    # exponential survival function (no event before sampling time), then sampling probability
                    result += -Λ * Δt + np.log(
                        self.ρ if node.event == "sampling" else 1 - self.ρ
                    )
                else:
                    # exponential density for event time
                    result += logΛ - Λ * Δt
                    # multinomial event probability
                    if node.event == "birth":
                        result += np.log(λ_x) - logΛ
                    elif node.event == "death":
                        result += np.log(self.μ) - logΛ
                    elif node.event == "mutation":
                        Δx = node.x - x
                        result += np.log(self.γ) - logΛ + norm.logpdf(Δx)
                    else:
                        raise ValueError(f"unknown event {node.event}")
        return result
