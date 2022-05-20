import jax.numpy as np

from gcdyn.model import Model
from gcdyn.parameters import Parameters


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

    params = Parameters(θ, μ, m, ρ)

    model = Model(params)
    model.simulate(T, 3)
    print(model.log_likelihood())


if __name__ == "__main__":
    main()
