r"""
Mutation effects generators :py:class:`Mutator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Abstract base class for defining generic mutation effect generators (i.e. :math:`\mathcal{p}(x\mid x')`),
with arbitrary :py:class:`ete3.TreeNode` attribute dependence.
Some concrete child classes are included.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Optional, Union, List
import pandas as pd
from scipy.stats import norm, gaussian_kde
import ete3

# NOTE: sphinx is currently unable to present this in condensed form, using a string type hint
# of "array-like" in the docstring args for now, instead of ArrayLike hint in call signature
# from numpy.typing import ArrayLike

class Mutator(ABC):
    r"""Abstract base class for generating mutation effects given
    :py:class:`ete3.TreeNode` object, which is modified in place."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def mutate(
        self,
        node: "ete3.TreeNode",
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        r"""Mutate a :py:class:`ete3.TreeNode` object in place.

        Args:
            node: A :py:class:`ete3.TreeNode` to mutate.
            seed: A seed to initialize the random number generation. If ``None``, then fresh,
                  unpredictable entropy will be pulled from the OS.
                  If an ``int``, then it will be used to derive the initial state.
                  If a :py:class:`numpy.random.Generator`, then that will be used directly.
        """

    @abstractmethod
    def logprob(self, node1: "ete3.TreeNode", node2: "ete3.TreeNode") -> float:
        r"""Compute the log probability that a mutation effect on ``node1``
        gives ``node2``.

        Args:
            node1: Initial node.
            node2: Mutant node.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in self.__dict__.items())})"


class AttrMutator(Mutator):
    r"""Abstract base class for mutators that mutate a specified
    :py:class:`ete3.TreeNode` node attribute.

    Args:
        attr: Node attribute to mutate.
    """

    @abstractmethod
    def __init__(self, attr: str = "x", *args: Any, **kwargs: Any) -> None:
        self.attr = attr

    def logprob(self, node1: "ete3.TreeNode", node2: "ete3.TreeNode") -> float:
        return self.prob(getattr(node1, self.attr), getattr(node2, self.attr), log=True)

    @abstractmethod
    def prob(self, attr1, attr2, log: bool = False) -> float:
        r"""Convenience method to compute the probability density (if ``attr``
        is continuous) or mass (if ``attr`` is discrete) that a mutation event
        brings attribute value ``attr1`` to attribute value ``attr2`` (e.g. for
        plotting).

        Args:
            attr1 (array-like): Initial attribute value.
            attr2 (array-like): Final attribute value.
            log: If ``True``, return the log probability density.
        """


class GaussianMutator(AttrMutator):
    r"""Gaussian mutation effects on a specified attribute.

    Args:
        shift: Mean shift wrt current attribute value.
        scale: Standard deviation of mutation effect.
        attr: Node attribute to mutate.
    """

    def __init__(
        self,
        shift: float = 0.0,
        scale: float = 1.0,
        attr: str = "x",
    ):
        super().__init__(attr=attr)
        self._distribution = norm(loc=shift, scale=scale)

    def mutate(
        self,
        node: "ete3.TreeNode",
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        new_value = getattr(node, self.attr) + self._distribution.rvs(random_state=seed)
        setattr(node, self.attr, new_value)

    def prob(self, attr1: float, attr2: float, log: bool = False) -> float:
        Δx = np.array(attr2) - np.array(attr1)
        return self._distribution.logpdf(Δx) if log else self._distribution.pdf(Δx)


class KdeMutator(AttrMutator):
    r"""Gaussian kernel density estimator (KDE) for mutation effect on a
    specified attribute.

    Args:
        dataset (array-like): Data to fit the KDE to.
        attr: Node attribute to mutate.
        bw_method: KDE bandwidth (see :py:class:`scipy.stats.gaussian_kde`).
        weights (optional array-like): Weights of data points (see :py:class:`scipy.stats.gaussian_kde`).
    """

    def __init__(
        self,
        dataset,
        attr: str = "x",
        bw_method: Optional[Union[str, float, callable]] = None,
        weights=None,
    ):
        super().__init__(attr=attr)
        self._distribution = gaussian_kde(dataset, bw_method, weights)

    def mutate(
        self,
        node: "ete3.TreeNode",
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        new_value = (
            getattr(node, self.attr)
            + self._distribution.resample(size=1, seed=seed)[0, 0]
        )
        setattr(node, self.attr, new_value)

    def prob(self, attr1: float, attr2: float, log: bool = False) -> float:
        Δx = np.array(attr2) - np.array(attr1)
        return self._distribution.logpdf(Δx) if log else self._distribution.pdf(Δx)


class DiscreteMutator(AttrMutator):
    r"""Mutations on a discrete space with a stochastic matrix.

    Args:
        state_space (array-like): hashable state values.
        transition_probs (array-like): Right-stochastic matrix, where column and row orders match the order of `state_space`.
        attr: Node attribute to mutate.
    """

    def __init__(
        self,
        state_space,
        transition_probs,
        attr: str = "x",
    ):
        transition_probs = np.array(transition_probs, dtype=float)
        if np.any(transition_probs < 0) or np.any(transition_probs.sum(axis=1) != 1):
            raise ValueError("Invalid stochastic matrix")

        super().__init__(attr=attr)

        self._transition_probs = pd.DataFrame(
            transition_probs, index=state_space, columns=state_space
        )

    def mutate(
        self,
        node: "ete3.TreeNode",
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:

        rng = np.random.default_rng(seed)

        transition_probs = self._transition_probs.loc[getattr(node, self.attr)]
        new_value = rng.choice(transition_probs.index, p=transition_probs)

        setattr(node, self.attr, new_value)

    def prob(self, attr1, attr2, log: bool = False) -> float:
        p = self._transition_probs.loc[attr1, attr2]
        return np.log(p) if log else p
