r"""
Mutation effects generators :py:class:`Mutator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Abstract base class for defining generic mutation effect generators (i.e. :math:`\mathcal{p}(x\mid x')`),
with arbitrary :py:class:`TreeNode` attribute dependence.
Some concrete child classes are included.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Optional, Union
from scipy.stats import norm, gaussian_kde
from gcdyn import bdms


class Mutator(ABC):
    r"""Abstract base class for generating mutation effects given
    :py:class:`TreeNode` object, which is modified in place."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def mutate(
        self,
        node: "bdms.TreeNode",
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        r"""Mutate a :py:class:`TreeNode` object in place.

        Args:
            node: A :py:class:`TreeNode` to mutate.
            seed: A seed to initialize the random number generation. If ``None``, then fresh,
                  unpredictable entropy will be pulled from the OS.
                  If an ``int``, then it will be used to derive the initial state.
                  If a :py:class:`numpy.random.Generator`, then that will be used directly.
        """

    @abstractmethod
    def logprob(self, node1: "bdms.TreeNode", node2: "bdms.TreeNode") -> float:
        r"""Compute the log probability that a mutation effect on ``node1``
        gives ``node2``.

        Args:
            node1: Initial node.
            node2: Mutant node.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in self.__dict__.items())})"


class PhenotypeMutator(Mutator):
    r"""Abstract base class for mutators that mutate a :py:class:`TreeNode` object's phenotype attribute
    :math:`x`.
    """

    def logprob(self, node1: "bdms.TreeNode", node2: "bdms.TreeNode") -> float:
        return self.probx(node1.x, node2.x, log=True)

    @abstractmethod
    def probx(self, x1: float, x2: float, log: bool = False) -> float:
        r"""Convenience method to compute the probability density (if :math:`x`
        is continuous) or mass (if :math:`x` is discrete) that a mutation event
        on phenotype :math:`x_1` gives phenotype :math:`x_2` (e.g. for
        plotting).

        Args:
            x1 (array-like): Initial phenotype value.
            x2 (array-like): Final phenotype value.
            log: If ``True``, return the log probability density.
        """


class GaussianMutator(PhenotypeMutator):
    r"""Gaussian mutation effect on phenotype attribute :math:`x`.

    Args:
        shift: Mean shift wrt current phenotype.
        scale: Standard deviation of mutation effect.
    """

    def __init__(self, shift: float = 0.0, scale: float = 1.0):
        self._distribution = norm(loc=shift, scale=scale)

    def mutate(
        self,
        node: "bdms.TreeNode",
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        node.x += self._distribution.rvs(random_state=seed)

    def probx(self, x1: float, x2: float, log: bool = False) -> float:
        Δx = np.array(x2) - np.array(x1)
        return self._distribution.logpdf(Δx) if log else self._distribution.pdf(Δx)


class KdeMutator(PhenotypeMutator):
    r"""Gaussian kernel density estimator (KDE) for mutation effect on phenotype
    attribute :math:`x`.

    Args:
        dataset (array-like): Data to fit the KDE to.
        bw_method: KDE bandwidth (see :py:class:`scipy.stats.gaussian_kde`).
        weights (optional array-like): Weights of data points (see :py:class:`scipy.stats.gaussian_kde`).
    """

    def __init__(
        self,
        dataset,
        bw_method: Optional[Union[str, float, callable]] = None,
        weights=None,
    ):
        self._distribution = gaussian_kde(dataset, bw_method, weights)

    def mutate(
        self,
        node: "bdms.TreeNode",
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        node.x += self._distribution.resample(size=1, seed=seed)[0, 0]

    def probx(self, x1: float, x2: float, log: bool = False) -> float:
        Δx = np.array(x2) - np.array(x1)
        return self._distribution.logpdf(Δx) if log else self._distribution.pdf(Δx)
