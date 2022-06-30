import jax.numpy as np
from jax import random

from gcdyn.model import Model
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

    n_trees = 2

    # # compare log likelihood before and after pruning trees
    # n_models = 3
    # model = Model(params)
    # for i in range(n_models):
    #     key, _ = random.split(key)
    #     model.simulate(T, n_trees, key[0])
    #     print("before:", model.log_likelihood())
    #     for tree in model.trees:
    #         tree.prune()
    #     print("after:", model.log_likelihood())

    model = Model(params)
    model.simulate(T, n_trees, seed)
    for tree in model.trees:
        print(len(tree.tree))
    θ_inferred = model.fit()
    print(θ_inferred)


if __name__ == "__main__":
    main()
