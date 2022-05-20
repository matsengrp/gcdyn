import jax.numpy as np
from jax import random

from gcdyn.tree import Tree
from gcdyn.parameters import Parameters


def main():
    T = 3
    seed = 0
    key = random.PRNGKey(seed)

    # response function parameters
    θ = np.array([3, 1, 0], dtype=float)
    # death rate
    μ = 1
    # mutation rate
    m = 1
    # sampling efficiency
    ρ = 0.5

    params = Parameters(θ, μ, m, ρ)

    n_trees = 10

    for i in range(n_trees):
        key, _ = random.split(key)
        tree = Tree(T, key, params)
        tree.draw_tree(f"tree {i + 1}.svg")


if __name__ == "__main__":
    main()
