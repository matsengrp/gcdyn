r"""BDMS inference.

.. todo::

    Harmonize/integrate with :py:mod:`gcdyn.bdms` module.
"""

import jax.numpy as np
from jax.scipy.stats import norm
from jax import jit
from jax.scipy.special import expit
from jaxopt import ScipyBoundedMinimize
from functools import partial

from gcdyn.bdms import TreeNode


class Model:
    r"""A class that represents a GC model.

    Args:
        μ (float): death rate
        γ (float): mutation rate
        ρ (float): sampling probability
    """

    def __init__(self, trees: list[TreeNode], μ: float, γ: float, ρ: float):
        self.trees = trees
        self.μ = μ
        self.γ = γ
        self.ρ = ρ

    def λ(self, x: float, θ):
        r"""Birth rate of phenotype x.

        Args:
            x: phenotype

        Returns:
            float: birth rate
        """
        return θ[0] * expit(θ[1] * (x - θ[2])) + θ[3]

    def fit(self, init_value = None, lower_bounds = None, upper_bounds = None, maxiter = 500):
        r"""Given a collection of :py:class:`TreeNode`, fit the parameters of
        the model."""

        def f(θ):
            return -self.log_likelihood(θ)

        optimizer = ScipyBoundedMinimize(fun=f, maxiter = maxiter)
        init_value = np.array(init_value, dtype = float) if init_value else np.array([2.0, 1.0, 0.0, 0.0])
        bounds = (
            np.array(lower_bounds, dtype = float) if lower_bounds else np.array([0, 0, -np.inf, 0.0]),
            np.array(upper_bounds, dtype = float) if upper_bounds else np.array([np.inf, np.inf, np.inf, np.inf]),
        )
        θ_inferred = optimizer.run(init_value, bounds)
        return θ_inferred

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, θ):
        r"""Compute log likelihood of parameters.

        Returns:
            float: log likelihood of the GC tree in the model
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
