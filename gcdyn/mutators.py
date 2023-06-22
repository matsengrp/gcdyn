r"""Mutation effects generators ^^^^^^^^^^^^^^^^^^^^^^^^^^^

Abstract base class for defining generic mutation effect generators
(i.e. :math:`\mathcal{p}(x\mid x')`), with arbitrary
:py:class:`ete3.TreeNode` attribute dependence. Some concrete child
classes are included.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Tuple
from collections.abc import Iterable, Callable
import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde
import ete3

from gcdyn.gpmap import GPMap
from gcdyn import utils

# NOTE: sphinx is currently unable to present this in condensed form when the sphinx_autodoc_typehints extension is enabled
from numpy.typing import ArrayLike


class Mutator(ABC):
    r"""Abstract base class for generating mutation effects given
    :py:class:`ete3.TreeNode` object, which is modified in place."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def mutate(
        self,
        node: ete3.TreeNode,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        r"""Mutate a :py:class:`ete3.TreeNode` object in place.

        Args:
            node: A :py:class:`ete3.TreeNode` to mutate.
            seed: A seed to initialize the random number generation. If ``None``, then fresh,
                  unpredictable entropy will be pulled from the OS.
                  If an ``int``, then it will be used to derive the initial state.
                  If a :py:class:`numpy.random.Generator`, then it will be used directly.
        """

    @abstractmethod
    def logprob(self, node: ete3.TreeNode) -> float:
        r"""Compute the log probability that a mutation effect on the parent of
        ``node`` gives ``node``.

        Args:
            node: Mutant node.
        """

    @property
    @abstractmethod
    def mutated_attrs(self) -> Tuple[str]:
        """Tuple of node attribute names that may be mutated by this
        mutator."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in vars(self).items() if not key.startswith('_'))})"

    def reset(self):
        pass


class AttrMutator(Mutator):
    r"""Abstract base class for mutators that mutate a specified
    :py:class:`ete3.TreeNode` node attribute.

    Args:
        attr: Node attribute to mutate.
    """

    def __init__(self, attr: str = "x", *args: Any, **kwargs: Any) -> None:
        self.attr = attr

    @property
    def mutated_attrs(self) -> Tuple[str]:
        return (self.attr,)

    def logprob(self, node: ete3.TreeNode) -> float:
        return self.prob(
            getattr(node.up, self.attr), getattr(node, self.attr), log=True
        )

    @abstractmethod
    def prob(self, attr1: ArrayLike, attr2: ArrayLike, log: bool = False) -> float:
        r"""Convenience method to compute the probability density (if ``attr``
        is continuous) or mass (if ``attr`` is discrete) that a mutation event
        brings attribute value ``attr1`` to attribute value ``attr2`` (e.g. for
        plotting).

        Args:
            attr1: Initial attribute value.
            attr2: Final attribute value.
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
        self.shift = shift
        self.scale = scale
        self._distribution = norm(loc=self.shift, scale=self.scale)

    def mutate(
        self,
        node: ete3.TreeNode,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        new_value = getattr(node, self.attr) + self._distribution.rvs(random_state=seed)
        setattr(node, self.attr, new_value)

    def prob(self, attr1: float, attr2: float, log: bool = False) -> float:
        Δx = np.asarray(attr2) - np.asarray(attr1)
        return self._distribution.logpdf(Δx) if log else self._distribution.pdf(Δx)


class KdeMutator(AttrMutator):
    r"""Gaussian kernel density estimator (KDE) for mutation effect on a
    specified attribute.

    Args:
        dataset: Data to fit the KDE to.
        attr: Node attribute to mutate.
        bw_method: KDE bandwidth (see :py:class:`scipy.stats.gaussian_kde`).
        weights: Weights of data points (see :py:class:`scipy.stats.gaussian_kde`).
    """

    def __init__(
        self,
        dataset: ArrayLike,
        attr: str = "x",
        bw_method: Optional[Union[str, float, Callable]] = None,
        weights: Optional[ArrayLike] = None,
    ):
        super().__init__(attr=attr)
        self._distribution = gaussian_kde(dataset, bw_method, weights)

    def mutate(
        self,
        node: ete3.TreeNode,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        new_value = (
            getattr(node, self.attr)
            + self._distribution.resample(size=1, seed=seed)[0, 0]
        )
        setattr(node, self.attr, new_value)

    def prob(self, attr1: float, attr2: float, log: bool = False) -> float:
        Δx = np.asarray(attr2) - np.asarray(attr1)
        return self._distribution.logpdf(Δx) if log else self._distribution.pdf(Δx)


class DiscreteMutator(AttrMutator):
    r"""Mutations on a discrete space with a stochastic matrix.

    Args:
        state_space: hashable state values.
        transition_matrix: Right-stochastic matrix, where column and row orders match the order of `state_space`.
        attr: Node attribute to mutate.
    """

    def __init__(
        self,
        state_space: Iterable,
        transition_matrix: ArrayLike,
        attr: str = "x",
    ):
        transition_matrix = np.asarray(transition_matrix, dtype=float)
        if not (
            transition_matrix.ndim == 2
            and transition_matrix.shape[0]
            == transition_matrix.shape[1]
            == len(state_space)
        ):
            raise ValueError(
                f"Transition matrix shape {transition_matrix.shape} does not match state space size {state_space}"
            )
        if np.any(transition_matrix < 0) or np.any(
            np.abs(transition_matrix.sum(axis=1) - 1) > 1e-4
        ):
            raise ValueError("Transition matrix is not a valid stochastic matrix.")

        super().__init__(attr=attr)

        self.state_space = {state: index for index, state in enumerate(state_space)}
        self.transition_matrix = transition_matrix

    def mutate(
        self,
        node: ete3.TreeNode,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        rng = np.random.default_rng(seed)

        states = list(self.state_space.keys())
        transition_probs = self.transition_matrix[
            self.state_space[getattr(node, self.attr)], :
        ]
        new_value = rng.choice(states, p=transition_probs)
        setattr(node, self.attr, new_value)

    def prob(self, attr1: float, attr2: float, log: bool = False) -> float:
        p = self.transition_matrix[self.state_space[attr1], self.state_space[attr2]]
        return np.log(p) if log else p


class SequenceMutator(AttrMutator):
    r"""Mutations on a DNA sequence.

    Nodes must have a ``sequence`` attribute
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(attr="sequence")

    def prob(self, attr1: float, attr2: float, log: bool = False) -> float:
        raise NotImplementedError


class UniformMutator(SequenceMutator):
    r"""Uniform mutation process.

    Args:
        node: An :py:class:`ete3.TreeNode` with a string-valued sequence attribute consisting of
              characters ``ACGT``.
        seed: See :py:class:`Mutator`.
    """

    def mutate(
        self,
        node: ete3.TreeNode,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        alphabet = "ACGT"
        rng = np.random.default_rng(seed)
        sequence = list(node.sequence)
        idx = rng.choice(len(sequence))
        sequence[idx] = rng.choice([base for base in alphabet if base != sequence[idx]])
        node.sequence = "".join(sequence)


class ContextMutator(SequenceMutator):
    """Class for hotspot-aware mutation model using mutability substitution
    probabilities expressed in terms of context.

    Args:
        mutability: Mutability values for each local nucleotide context.
        substitution: Table of nucleotide substitution bias (columns) for each local nucleotide context (index).
    """

    def __init__(
        self,
        mutability: pd.Series,
        substitution: pd.DataFrame,
    ):
        super().__init__()
        self.mutability = mutability.to_dict()
        self.substitution = substitution.fillna(0.0).T.to_dict()
        self.cached_ctx_muts = {}

    def reset(self):
        """Clear cached context mutabilities"""
        self.cached_ctx_muts.clear()

    def mutate(
        self,
        node: ete3.TreeNode,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Mutate ``node.sequence`` according to an AID hotspot-aware mutation
        model using mutability values at each nucleotide 5mer and substitution
        probabilities.

        Args:
            node: node with sequence, consisting of characters ``ACGT``
            seed: See :py:class:`Mutator`.
        """
        rng = np.random.default_rng(
            seed
        )  # would be better not to create a new rng for every call
        seq_contexts = utils.node_contexts(node)
        if node.sequence not in self.cached_ctx_muts:
            self.cached_ctx_muts[node.sequence] = np.asarray(
                [self.mutability[context] for context in seq_contexts]
            )
        mutabilities = self.cached_ctx_muts[node.sequence]
        i = rng.choice(len(mutabilities), p=mutabilities / mutabilities.sum())
        context = seq_contexts[i]
        sub_nt = rng.choice(
            list(self.substitution[context].keys()),
            p=list(self.substitution[context].values()),
        )
        node.sequence = node.sequence[:i] + sub_nt + node.sequence[(i + 1) :]

    @property
    def mutated_attrs(self) -> Tuple[str]:
        return super().mutated_attrs + ("chain_2_start_idx",)


class SequencePhenotypeMutator(AttrMutator):
    r"""Mutations on a DNA sequence that get translated into a functional
    phenotype.

    Args:
        sequence_mutator: A SequenceMutator object.
        gp_map: a map from sequence to phenotype that will be used after applying a
                sequence-level mutation.
        attr: Node attribute to update using the gp_map after sequence mutation.
    """

    def __init__(
        self,
        sequence_mutator: SequenceMutator,
        gp_map: Optional[GPMap] = None,
        attr: str = "x",
    ):
        self.sequence_mutator = sequence_mutator
        self.gp_map = gp_map
        super().__init__(attr=attr)

    def reset(self):
        """Clear cached context mutabilities in sequence_mutator"""
        self.sequence_mutator.reset()

    def mutate(
        self,
        node: ete3.TreeNode,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        if self.gp_map is not None:
            current_attr = getattr(node, self.attr)
            assert current_attr == self.gp_map(
                node.sequence
            ), "Unexpected phenotype given sequence. Did you forget to initialize the phenotype?"

        self.sequence_mutator.mutate(node, seed)

        if self.gp_map is not None:
            setattr(node, self.attr, self.gp_map(node.sequence))

    def prob(self, attr1, attr2, log: bool = False) -> float:
        raise NotImplementedError(
            "This doesn't make sense according to the current "
            "formulation because attr1 and attr2 will be phenotypes."
        )

    @property
    def mutated_attrs(self) -> Tuple[str]:
        return (self.attr,) + self.sequence_mutator.mutated_attrs
