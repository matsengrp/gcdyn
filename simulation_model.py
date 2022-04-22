import jax.numpy as np

import gc_model
import parameters


def main():
    T = 3

    # response function parameters
    θ = np.array([3, 1, 0], dtype=float)
    # death rate
    μ = 1
    # mutation rate
    m = 1
    # sampling efficiency
    ρ = 0.5

    params = parameters.Parameters(θ, μ, m, ρ)

    model = gc_model.GC_model(params)
    model.simulate(T, 3)
    print(model.log_likelihood())


if __name__ == "__main__":
    main()
