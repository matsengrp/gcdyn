import jax.numpy as np

from dataclasses import dataclass


@dataclass
class Parameters:
    r"""A dataclass that represents model parameter for GC tree and GC forest

    Args:
        θ (np.array): response function parameters used to find birth rate
        μ (float): death rate
        m (float): mutation rate
        ρ (float): sampling efficiency
    """
    def __init__(self, θ: np.array, μ: float, m: float, ρ: float):
        self.θ = θ
        self.μ = μ
        self.m = m
        self.ρ = ρ
