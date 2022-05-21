r"""Model parameter classes"""

import jax.numpy as np

from dataclasses import dataclass


@dataclass
class Parameters:
    r"""A dataclass that represents model parameter for GC tree and GC forest

    Attributes:
        θ: response function parameters used to find birth rate
        μ: death rate
        m: mutation rate
        ρ: sampling efficiency
    """
    θ: np.array
    μ: float
    m: float
    ρ: float
