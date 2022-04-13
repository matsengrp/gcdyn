import jax.numpy as np
from jax import random
from jax.scipy.stats import norm

import gc_tree
import parameters


class GC_model:
    def __init__(self, params):
        self.params = params
        self.trees = None

    def simulate(self, T, n_trees):
        """
        Creates a collection of GC_tree given a key
        """
        # call GC_tree constructor
        # evolve the tree -- call method in tree class
        seed = 0
        key = random.PRNGKey(seed)
        trees = []
        for i in range(n_trees):
            key, _ = random.split(key)
            tree = gc_tree.GC_tree(T, key, self.params)
            trees.append(tree)
        self.trees = trees

    def log_likelihood(self):
        """
        Find log likelihood of simulated trees
        """
        result = 0
        for tree in self.trees:
            for node in tree.tree.children[0].traverse():
                x = node.up.x
                Δt = node.dist
                λ_x = tree.λ(x)
                Λ = λ_x + self.params.μ + self.params.m
                logΛ = np.log(Λ)
                if node.event in ("sampled", "unsampled"):
                    # exponential survival function (no event before sampling time), then sampling probability
                    result += -Λ * Δt + np.log(
                        self.params.ρ if node.event == "sampled" else 1 - self.params.ρ
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

    def fit(self):
        """
        Given a collection of GC_trees, fit the parameters of the model
        """
        pass
