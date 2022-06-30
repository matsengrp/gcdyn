r"""Model classes"""

import jax.numpy as np
from jax import random
from jax.scipy.stats import norm
from jax import jit, grad
from jax.scipy.special import expit
import ete3
import matplotlib as mp
import matplotlib.pyplot as plt
import mushi.optimization as opt

from gcdyn.tree import Tree

from gcdyn.parameters import Parameters


class Model:
    r"""A class that represents a GC model

    Args:
        params: model parameters
    """

    def __init__(self, params: Parameters):
        self.params = params
        self.trees = None

    def λ(self, x: float, θ):
        r"""Birth rate of phenotype x

        Args:
            x: phenotype

        Returns:
            float: birth rate
        """
        return θ[0] * expit(θ[1] * (x - θ[2]))

    def simulate(self, T: float, n_trees: int, seed: int):
        r"""Creates a collection of ``Tree`` given a key.

        Args:
            T: simulation sampling time
            n_trees: number of GC trees in the model
            seef: random seed
        """
        key = random.PRNGKey(seed)
        trees = []
        for i in range(n_trees):
            while True:
                key, _ = random.split(key)
                tree = Tree(T, key[0], self.params)
                self._evolve(tree.tree, T, key)
                if 50 < len(tree.tree) < 96:
                    trees.append(tree)
                    break
        self.trees = trees

    def _evolve(self, tree: ete3.Tree, t: float, key: np.ndarray):
        r"""Evolve an ETE Tree node with a phenotype attribute for time t

        Args:
            tree:initial tree to evolve
            t: sampling time
            key: random key, i.e. generated with :py:func:`jax.random.PRNGKey()` or :py:func:`jax.random.split()`
        """
        λ_x = self.λ(tree.x, self.params.θ)
        Λ = λ_x + self.params.μ + self.params.m
        time_key, event_key = random.split(key)
        τ = random.exponential(time_key) / Λ
        if τ > t:
            child = ete3.Tree(name=tree.name + 1, dist=t)
            child.x = tree.x
            child.t = tree.t + t
            child.event = (
                "sampled" if random.uniform(event_key) < self.params.ρ else "unsampled"
            )
            tree.add_child(child)
            return

        possible_events = ["birth", "death", "mutation"]
        event_probabilities = np.array([λ_x, self.params.μ, self.params.m]) / Λ
        event = possible_events[
            random.choice(event_key, len(possible_events), p=event_probabilities)
        ]
        child = ete3.Tree(name=tree.name + 1, dist=τ)
        child.t = tree.t + τ
        child.x = tree.x
        child.event = event
        if event == "birth":
            child1_key, child2_key = random.split(event_key)
            self._evolve(child, t - τ, child1_key)
            self._evolve(child, t - τ, child2_key)
        elif event == "death":
            pass
        elif event == "mutation":
            mutation_key, child_key = random.split(event_key)
            child.x += random.normal(mutation_key)
            self._evolve(child, t - τ, child_key)
        else:
            raise ValueError("unknown event")
        tree.add_child(child)

    def fit(self):
        r"""Given a collection of `tree.Tree`, fit the parameters of the model."""

        @jit
        def log_likelihood(θ):
            r"""Find log likelihood of simulated trees.

            Returns:
                float: log likelihood of the GC tree in the model
            """
            result = 0
            for tree in self.trees:
                for node in tree.tree.children[0].traverse():
                    x = node.up.x
                    Δt = node.dist
                    λ_x = self.λ(x, θ)
                    Λ = λ_x + self.params.μ + self.params.m
                    logΛ = np.log(Λ)
                    if node.event in ("sampled", "unsampled"):
                        # exponential survival function (no event before sampling time), then sampling probability
                        result += -Λ * Δt + np.log(
                            self.params.ρ
                            if node.event == "sampled"
                            else 1 - self.params.ρ
                        )
                    else:
                        # exponential density for event time
                        result += logΛ - Λ * Δt
                        # multinomial event probability
                        if node.event == "birth":
                            result += np.log(λ_x) - logΛ
                        elif node.event == "death":
                            result += np.log(self.params.μ) - logΛ
                        elif node.event == "mutation":
                            Δx = node.x - x
                            result += np.log(self.params.m) - logΛ + norm.logpdf(Δx)
                        else:
                            raise ValueError(f"unknown event {node.event}")
            return result

        @jit
        def g(θ):
            return -log_likelihood(θ)

        @jit
        def h(θ):
            return 0

        @jit
        def prox(θ, s):
            return np.clip(θ, np.array([1e-1, -np.inf, -np.inf]))

        grad_g = jit(grad(g))
        optimizer = opt.AccProxGrad(g, grad_g, h, prox, verbose=True)
        θ_inferred = optimizer.run(np.array([3.0, 1.0, 0.0]), max_iter=1000, tol=0)
        return θ_inferred
