import jax.numpy as np
from jax import random

import gc_tree


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

    n_trees = 10

    for i in range(n_trees):
        key, _ = random.split(key)
        tree = gc_tree.GC_tree(T, key, θ, μ, m, ρ)
        tree.draw_tree(f"tree {i + 1}.svg")


if __name__ == "__main__":
    main()
