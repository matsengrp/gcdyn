r"""Germinal center light zone / dark-zone cycles simulator.

Borrowing from `gctree <https://github.com/matsengrp/gctree/blob/master/gctree/mutation_model.py>`_
"""

from collections.abc import Iterable
from typing import Callable, List, Any
from ete3 import TreeNode


class GC:
    r"""A class for simulating a germinal center with discrete LZ :math:`\leftrightarrow` DZ cycles.

    Args:
        sequence: root nucleotide sequence
        dz_subtree_simulator: neutral tree generator for DZ proliferation and mutation. This tree is assumed to run for exactly one unit of simulation time, but this is not enforced (if not, you will generate non-ultrametric trees). If it produces dead leaves, they should be marked with a node attribute ``node.terminated = True``.
        fitness_function: takes a list of sequences and returns their fitness parameters
        N0: initial naive abundance
    """

    def __init__(
        self,
        sequence: str,
        dz_subtree_simulator: Callable[[TreeNode, float], None],
        fitness_function: Callable[
            [
                Iterable[str],
            ],
            List[Any],
        ],
        N0: int = 1,
    ):
        self.tree = TreeNode(dist=0)
        self.tree.sequence = sequence
        self.tree.terminated = False
        self.dz_subtree_simulator = dz_subtree_simulator
        self.fitness_function = fitness_function

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
        fitnesses = self.fitness_function([leaf.sequence for leaf in alive_leaves])
        for leaf, args in zip(alive_leaves, fitnesses):
            self.dz_subtree_simulator(leaf, *args)

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
