r"""Germinal center light zone / dark-zone cycles simulator.

Borrowing from `gctree <https://github.com/matsengrp/gctree/blob/master/gctree/mutation_model.py>`_
"""

from typing import Callable, List, Any
from ete3 import TreeNode
import numpy as np
from numpy.random import default_rng


class GC:
    r"""A class for simulating a germinal center with discrete LZ :math:`\leftrightarrow` DZ cycles.

    Args:
        sequence: root nucleotide sequence
        proliferator: neutral tree generator for DZ proliferation and mutation. This tree is assumed to run for exactly one unit of simulation time, but this is not enforced (if not, you may generate non-ultrametric trees). If it produces dead leaves, they should be marked with a node attribute ``node.terminated = True``.
        mutator: mutation generator that takes a starting sequence and an exposure time and returns a mutated sequence
        selector: takes a list of sequences and returns their fitness parameters
        N0: initial naive abundance
    """

    def __init__(
        self,
        sequence: str,
        proliferator: Callable[[TreeNode, float, np.random.Generator], None],
        mutator: Callable[[str, float, np.random.Generator], str],
        selector: Callable[[List[str]], List[Any]],
        N0: int = 1,
    ):
        self.tree = TreeNode(dist=0)
        self.tree.sequence = sequence
        self.tree.terminated = False
        self.proliferator = proliferator
        self.mutator = mutator
        self.selector = selector

        if N0 > 1:
            for _ in range(N0):
                child = TreeNode()
                child.dist = 0
                child.sequence = sequence
                child.terminated = False
                self.tree.add_child(child)

    def step(self) -> None:
        r"""Simulate one cycle."""
        alive_leaves = [leaf for leaf in self.tree if not leaf.terminated]
        fitnesses = self.selector([leaf.sequence for leaf in alive_leaves])
        for leaf, args in zip(alive_leaves, fitnesses):
            self.proliferator(leaf, *args)
            if self.mutator:
                for node in leaf.iter_descendants():
                    node.sequence = self.mutator(node.up.sequence, node.dist)

    def prune(self) -> None:
        r"""Prune the tree to the subtree induced by the alive leaves."""
        event_cache = self.tree.get_cached_content(store_attr="terminated")

        def is_leaf_fn(node):
            return False not in event_cache[node]

        if is_leaf_fn(self.tree):
            raise RuntimeError("dead")

        for node in self.tree.traverse(is_leaf_fn=is_leaf_fn):
            if is_leaf_fn(node):
                node.detach()

    def simulate(self, T: int, prune: bool = True) -> None:
        r"""Simulate.

        Args:
            T: number of cycles to simulate
            prune: prune to the tree induced by the surviving lineages
        """
        for _ in range(T):
            self.step()
        if prune:
            self.prune()


def binary_proliferator(
    treenode: TreeNode, p: float, rng: np.random.Generator = default_rng()
):
    r"""binary dark zone simulation (Galton-Watson)

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
