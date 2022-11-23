r"""
Mutation effects generators
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Abstract base class for defining generic mutation effect generators (i.e. :math:`\mathcal{p}(x\mid x')`),
with arbitrary :py:class:`ete3.TreeNode` attribute dependence.
Some concrete child classes are included.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde
import ete3
from gcdyn.gpmap import GPMap

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
        node: ete3.TreeNode,
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
    def logprob(self, node: ete3.TreeNode) -> float:
        r"""Compute the log probability that a mutation effect on the parent of
        ``node`` gives ``node``.

        Args:
            node: Mutant node.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in vars(self).items() if not key.startswith('_'))})"


class AttrMutator(Mutator):
    r"""Abstract base class for mutators that mutate a specified
    :py:class:`ete3.TreeNode` node attribute.

    Args:
        attr: Node attribute to mutate.
    """

    @abstractmethod
    def __init__(self, attr: str = "x", *args: Any, **kwargs: Any) -> None:
        self.attr = attr

    def logprob(self, node: ete3.TreeNode) -> float:
        return self.prob(
            getattr(node.up, self.attr), getattr(node, self.attr), log=True
        )

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
        node: ete3.TreeNode,
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
        transition_matrix (array-like): Right-stochastic matrix, where column and row orders match the order of `state_space`.
        attr: Node attribute to mutate.
    """

    def __init__(
        self,
        state_space,
        transition_matrix,
        attr: str = "x",
    ):
        transition_matrix = np.array(transition_matrix, dtype=float)
        if np.any(transition_matrix < 0) or np.any(
            np.abs(transition_matrix.sum(axis=1) - 1) > 1e-4
        ):
            raise ValueError("Invalid stochastic matrix")

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

    def prob(self, attr1, attr2, log: bool = False) -> float:
        p = self.transition_matrix[self.state_space[attr1], self.state_space[attr2]]
        return np.log(p) if log else p


class SequenceMutator(AttrMutator):
    r"""Mutations on a DNA sequence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(attr="sequence")

    def prob(self, attr1, attr2, log: bool = False) -> float:
        raise NotImplementedError


class UniformMutator(SequenceMutator):
    r"""Uniform mutation process.

    Args:
        node: a TreeNode with a string-valued sequence attribute consisting of
              characters ``ACGT``
        seed: See above.
    """

    def mutate(
        self,
        node: "ete3.TreeNode",
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
        mutability: a mapping from local context to mutability
        substitution: a mapping from local context to substitution process
        seq_to_contexts: a function that accepts a sequence and splits it into local contexts
    """

    def __init__(
        self,
        mutability: pd.Series,
        substitution: pd.DataFrame,
        seq_to_contexts: Callable[str, List[str]],
    ):
        super().__init__()

        self.mutability = mutability
        self.substitution = substitution
        self.seq_to_contexts = seq_to_contexts

    def mutate(
        self,
        node: "ete3.TreeNode",
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Mutate node.sequence according to an AID hotspot-aware mutation model using
        mutability values at each nucleotide 5mer and substitution probabilities.

        Args:
            node: node with sequence, consisting of characters ``ACGT``
            seed: See above.
        """

        rng = np.random.default_rng(seed)

        contexts = self.seq_to_contexts(node.sequence)
        mutabilities = np.array([self.mutability[context] for context in contexts])
        i = rng.choice(len(mutabilities), p=mutabilities / sum(mutabilities))
        sub_nt = rng.choice(
            self.substitution.columns,
            p=self.substitution.loc[contexts[i]].fillna(0),
        )
        node.sequence = node.sequence[:i] + sub_nt + node.sequence[(i + 1) :]


class SequencePhenotypeMutator(AttrMutator):
    r"""Mutations on a DNA sequence that get translated into a functional phenotype.

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

    def mutate(
        self,
        node: "ete3.TreeNode",
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
        raise NotImplementedError
