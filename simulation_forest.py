import jax.numpy as np
from jax import random

import gc_forest


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

    forest = gc_forest.GC_forest(T, key, θ, μ, m, ρ, n_trees)


if __name__ == "__main__":
    main()
