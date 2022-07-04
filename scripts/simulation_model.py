import jax.numpy as np
from jax import random

from gcdyn.model import Model
from gcdyn.parameters import Parameters


def main():
    T = 3
    seed = 0

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

    model = Model(params)
    model.simulate(T, n_trees, seed)
    for tree in model.trees:
        print(len(tree.tree))
    θ_inferred = model.fit()
    print(θ_inferred)


if __name__ == "__main__":
    main()
