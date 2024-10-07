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

# TODO rm this path stuff and pip install to go back to using installed bdms-sim (rather than cloned github copy)
#   - pip install bdms-sim
#   - pip uninstall bdms-sim
import os
import sys
bdms_dir = os.path.dirname(os.path.realpath(__file__)).replace('/gcdyn/gcdyn', '/bdms')
sys.path.insert(1, bdms_dir)
from bdms.mutators import Mutator as BDMSMutator

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
    def node_attrs(self) -> Tuple[str]:
        """Tuple of node attribute names that need to be propagated to child nodes."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in vars(self).items() if not key.startswith('_'))})"

    def cleanup(self):
        """Perform any cleanup required when finishing the simulation of each tree."""
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
    def node_attrs(self) -> Tuple[str]:
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


class SequenceMutator(BDMSMutator):
    r"""Mutations on a DNA sequence.

    Nodes must have a ``sequence`` attribute
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(attr="sequence")

    @property
    def node_attrs(self) -> Tuple[str]:
        return (self.attr,)

    def check_node(self, node: ete3.TreeNode):
        for tattr in self.node_attrs:
            if not hasattr(node.state, tattr):
                raise Exception("required attributed '%s' not found on node" % tattr)

    def prob(self, attr1: float, attr2: float, log: bool = False) -> float:
        raise NotImplementedError


# class UniformMutator(SequenceMutator):
#     r"""Uniform mutation process.

#     Args:
#         node: An :py:class:`ete3.TreeNode` with a string-valued sequence attribute consisting of
#               characters ``ACGT``.
#         seed: See :py:class:`Mutator`.
#     """

#     def mutate(
#         self,
#         node: ete3.TreeNode,
#         seed: Optional[Union[int, np.random.Generator]] = None,
#     ) -> None:
#         self.check_node(node)
#         alphabet = "ACGT"
#         rng = np.random.default_rng(seed)
#         sequence = list(node.sequence)
#         idx = rng.choice(len(sequence))
#         sequence[idx] = rng.choice([base for base in alphabet if base != sequence[idx]])
#         node.sequence = "".join(sequence)


class ContextMutator(SequenceMutator):
    """Class to mutate a node's sequence using a hotspot-aware mutation model with mutability substitution
    probabilities expressed in terms of context.

    Args:
        mutability: Mutability values for each local nucleotide context.
        substitution: Table of nucleotide substitution bias (columns) for each local nucleotide context (index).
    """

    def __init__(
        self,
        mutability: pd.Series,
        substitution: pd.DataFrame,
        attr: str = "sequence",
    ):
        super().__init__()
        self.mutability = mutability.to_dict()
        self.substitution = substitution.fillna(0.0).T.to_dict()
        self.cached_ctx_muts = {}
        super().__init__(attr=attr)

    def cleanup(self):
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
        self.check_node(node)
        rng = np.random.default_rng(seed)
        seq_contexts = utils.nodestate_contexts(node.state)
        if node.state.sequence not in self.cached_ctx_muts:
            self.cached_ctx_muts[node.state.sequence] = np.asarray(
                [self.mutability[context] for context in seq_contexts]
            )
        mutabilities = self.cached_ctx_muts[node.state.sequence]
        i = rng.choice(len(mutabilities), p=mutabilities / mutabilities.sum())
        context = seq_contexts[i]
        sub_nt = rng.choice(
            list(self.substitution[context].keys()),
            p=list(self.substitution[context].values()),
        )
        node.state.sequence = node.state.sequence[:i] + sub_nt + node.state.sequence[(i + 1) :]

    @property
    def node_attrs(self) -> Tuple[str]:
        return super().node_attrs + ("chain_2_start_idx",)


class SequencePhenotypeMutator(BDMSMutator):
    r"""Class to mutate a node's sequence, and then translate that sequence
    modification into a modification of the node's phenotype (attribute).

    Args:
        sequence_mutator: A SequenceMutator object.
        gp_map: a map from sequence to phenotype that will be used after applying a
                sequence-level mutation.
        attr: Node attribute to update (for synchronization with birth/death processes, and maybe not really needed any more if they're all just 'state')
        phen_attr: phenotype attribute
    """

    def __init__(
        self,
        sequence_mutator: SequenceMutator,
        gp_map: Optional[GPMap] = None,
        attr: str = "state",
        phen_attr: str = 'x',
    ):
        self.sequence_mutator = sequence_mutator
        self.gp_map = gp_map
        self.phen_attr = phen_attr
        super().__init__(attr=attr)

    def cleanup(self):
        """Clear cached context mutabilities in sequence_mutator"""
        self.sequence_mutator.cleanup()

    def mutate(
        self,
        node: ete3.TreeNode,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        if self.gp_map is not None:
            current_attr = getattr(node.state, self.phen_attr)
            assert current_attr == self.gp_map(
                node.state.sequence
            ), "Unexpected phenotype given sequence. Did you forget to initialize the phenotype?"

        self.sequence_mutator.mutate(node, seed)

        if self.gp_map is not None:
            setattr(node.state, self.phen_attr, self.gp_map(node.state.sequence))

    def prob(self, attr1, attr2, log: bool = False) -> float:
        raise NotImplementedError(
            "This doesn't make sense according to the current "
            "formulation because attr1 and attr2 will be phenotypes."
        )

    @property
    def node_attrs(self) -> Tuple[str]:
        return (self.attr,) + self.sequence_mutator.node_attrs
