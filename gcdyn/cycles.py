r"""Germinal center light zone / dark-zone cycles simulator.

Borrowing from `gctree <https://github.com/matsengrp/gctree/blob/master/gctree/mutation_model.py>`_
"""

from typing import Callable, List, Any, Optional
from ete3 import TreeNode
import numpy as np
from numpy.random import default_rng


class ExtinctionError(Exception):
    """The simulation has resulted in extintion of all lineages."""


class GC:
    r"""A class for simulating a germinal center with discrete LZ :math:`\leftrightarrow` DZ cycles.

    Args:
        sequence: root nucleotide sequence
        proliferator: neutral tree generator for DZ proliferation and mutation. This tree is assumed to run for exactly one unit of simulation time, but this is not enforced (if not, you may generate non-ultrametric trees). If it produces extinct leaves, they should be marked with a node attribute ``node.terminated = True``.
        mutator: mutation generator that takes a starting sequence and an exposure time and returns a mutated sequence
        selector: takes a list of sequences and returns their fitness parameters
        N0: initial naive abundance
        Nmax: population capacity. If not ``None`` and the number of alive cells exceeds ``Nmax`` in any cycle, the population will be randomly downsampled to size ``Nmax``.
        rng: random number generator
    """

    def __init__(
        self,
        sequence: str,
        proliferator: Callable[[TreeNode, float, np.random.Generator], None],
        mutator: Callable[[str, float, np.random.Generator], str],
        selector: Callable[[List[str]], List[Any]],
        N0: int = 1,
        Nmax: Optional[int] = None,
        rng: np.random.Generator = default_rng(),
    ):
        self.tree = TreeNode(dist=0)
        self.tree.sequence = sequence
        self.tree.terminated = False
        self.proliferator = proliferator
        self.mutator = mutator
        self.selector = selector
        self.Nmax = Nmax
        self.rng = rng

        if N0 > 1:
            for _ in range(N0):
                child = TreeNode()
                child.dist = 0
                child.sequence = sequence
                child.terminated = False
                self.tree.add_child(child)

        self.alive_leaves = set([leaf for leaf in self.tree])

    def step(self, enforce_timescale: bool = True) -> None:
        r"""Simulate one cycle.

        Args:
            enforce_timescale: if ``True``, the time scale of the DZ proliferation tree must be consistent with the global timescale (one unit per step)
        """
        fitnesses = self.selector([leaf.sequence for leaf in self.alive_leaves])
        for leaf, args in zip(self.alive_leaves, fitnesses):
            self.proliferator(leaf, *args, rng=self.rng)
            for node in leaf.iter_descendants():
                node.sequence = self.mutator(node.up.sequence, node.dist, rng=self.rng)
            if enforce_timescale:
                for subleaf in leaf:
                    if not subleaf.terminated and leaf.get_distance(subleaf) != 1:
                        raise ValueError(
                            "DZ subtree timescale is not consistent with simulation time"
                        )
        self.alive_leaves = set([leaf for leaf in self.tree if not leaf.terminated])

        if self.Nmax:
            for leaf in self.rng.choice(
                list(self.alive_leaves),
                size=max(0, len(self.alive_leaves) - self.Nmax),
                replace=False,
            ):
                leaf.terminated = True
                self.alive_leaves.remove(leaf)

    def prune(self) -> None:
        r"""Prune the tree to the subtree induced by the alive leaves."""
        event_cache = self.tree.get_cached_content(store_attr="terminated")

        def is_leaf_fn(node):
            return False not in event_cache[node]

        if is_leaf_fn(self.tree):
            raise ExtinctionError()

        for node in self.tree.traverse(is_leaf_fn=is_leaf_fn):
            if is_leaf_fn(node):
                node.detach()

    def simulate(
        self,
        T: int,
        prune: bool = True,
        max_tries: int = 100,
        enforce_timescale: bool = True,
    ) -> None:
        r"""Simulate.

        Args:
            T: number of cycles to simulate
            prune: prune to the tree induced by the surviving lineages
            max_tries: try this many times to simulate a tree that doesn't go extinct
            enforce_timescale: if ``True``, the time scale of the DZ proliferation trees must be consistent with the global timescale
        """
        success = False
        n_tries = 0
        while not success:
            try:
                for _ in range(T):
                    self.step(enforce_timescale=enforce_timescale)
                if prune:
                    self.prune()
                success = True
            except ExtinctionError as err:
                n_tries += 1
                if n_tries == max_tries:
                    raise err
                for child in self.tree.children:
                    self.tree.remove_child(child)
                self.tree.terminated = False


def binary_proliferator(
    treenode: TreeNode, p: float, rng: np.random.Generator = default_rng()
):
    r"""Binary dark zone step (Galton-Watson).
    With probability :math:`1-p` the input's ``terminated`` attribute is set to ``True``, indicating extinction.
    With probability :math:`p` add two children to the the input with ``terminated`` attributes set to ``False``, indicating birth.

    Args:
        treenode: root node to simulate from
        p: bifurcation probability :math:`p \in [0, 1]`
        rng: random number generator
    """
    assert 0 <= p <= 1
    if rng.random() > p:
        treenode.terminated = True
    else:
        for _ in range(2):
            child = TreeNode()
            child.dist = 1
            child.sequence = treenode.sequence
            child.terminated = False
            treenode.add_child(child)


def uniform_mutator(
    sequence: str, time: float, rng: np.random.Generator = default_rng()
):
    r"""Uniform mutation process at rate 1.

    Args:
        sequence: initial sequence, consisting of characters ``ACGT``
        time: exposure time
        rng: random number generator
    """
    alphabet = "ACGT"
    sequence = list(sequence)
    t = 0
    while True:
        t += rng.exponential()
        if t < time:
            idx = rng.choice(len(sequence))
            sequence[idx] = rng.choice(
                [base for base in alphabet if base != sequence[idx]]
            )
        else:
            break
    return "".join(sequence)
