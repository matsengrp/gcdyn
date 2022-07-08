r"""Germinal center light zone / dark-zone cycles simulator.

Borrowing from `gctree <https://github.com/matsengrp/gctree/blob/master/gctree/mutation_model.py>`_
"""

from typing import Callable, List, Any, Optional, Tuple
from ete3 import TreeNode
import numpy as np
from numpy.random import default_rng
from gcdyn.fitness import Fitness
from gcdyn.replay import ReplayPhenotype
from math import floor, ceil


class ExtinctionError(Exception):
    """The simulation has resulted in extinction of all lineages."""


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
                    if (
                        not subleaf.terminated
                        and subleaf != leaf
                        and round(leaf.get_distance(subleaf)) != 1
                    ):
                        raise ValueError(
                            "DZ subtree timescale is not consistent with simulation time, val = {0}".format(
                                leaf.get_distance(subleaf)
                            )
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
            enforce_timescale: if ``True``, the timescale of the DZ proliferation trees must be consistent with the global timescale
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
    With probability :math:`p` add two children to the input with ``terminated`` attributes set to ``False``, indicating birth.

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


def uniform_selector(sequence_list) -> List[Tuple]:
    """Uniform selector assigning normalized fitness of each sequence to
    1/`len(sequence_list)`

    Args:
        sequence_list: list of sequences for fitness assignment

    Returns:
        fitness_values: list of tuples containing the assigned fitness (equal values)
    """
    fitness_values = [tuple([1 / len(sequence_list)]) for seq in sequence_list]
    return fitness_values


def replay_cell_div_selector(
    sequence_list: list[str], slope: float = 3.47, y_intercept: float = 1.3
) -> List[Tuple]:
    r"""Determines the number of cell divisions based on a list of sequences using Replay

    Args:
        sequence_list: list of nucleotide sequences to use to predict the number of cell divisions

    Returns:
        cell_divs: list of tuples containing the predicted number of cell divisions
    """

    sig_fit = Fitness(Fitness.sigmoidal_fitness)
    replay_phenotype = ReplayPhenotype(
        1,
        1,
        336,
        "https://raw.githubusercontent.com/jbloomlab/Ab-CGGnaive_DMS/main/data/CGGnaive_sites.csv",
        "Linear.model",
        ["delta_log10_KD", "expression"],
        -10.43,
    )
    sig_fit_df = sig_fit.normalized_fitness_df(
        sequence_list, calculate_KD=replay_phenotype.calculate_KD
    )
    cell_divs = [
        tuple([cell_div])
        for cell_div in sig_fit.cell_divisions_from_tfh_linear(
            sig_fit_df, slope, y_intercept
        )
    ]
    return cell_divs


def simple_proliferator(
    treenode: TreeNode,
    cell_divisions: int,
    dist: float = None,
    rng: np.random.Generator = default_rng(),
) -> None:
    r"""Recursively populates descendants on a tree node based on the number of integer cell divisions indicated.
    Branch lengths are set to 1/`cell_divisions`, such that the distance from the root to all leaves will be 1.

    Args:
        treenode: root node to populate from
        cell_divisions: number of cell divisions from the cell represented by ``treenode``
        dist: distance from each parent to child node
        rng: random number generator
    """
    if dist is None:
        dist = 1 / cell_divisions
    if cell_divisions > 0:
        for _ in range(2):
            child = TreeNode()
            child.dist = dist
            child.sequence = treenode.sequence
            child.terminated = False
            treenode.add_child(child)
            simple_proliferator(child, cell_divisions - 1, dist)


def cell_div_balanced_proliferator(
    treenode: TreeNode, cell_divisions: float, rng: np.random.Generator = default_rng()
) -> None:
    r"""Populates descendants on a tree node based on the number of cell divisions indicated, producing the number of children expected by :math:`2^cell_divisions`, where ``cell_divisions`` is not necessarily an integer value.
    A full binary tree is generated, then leaf nodes are removed at random from the set of alternating leaf nodes until there are the expected number of children.

    Args:
        treenode: root node to populate from
        cell_divisions: number of cell divisions from the cell represented by ``treenode``
        rng: random number generator
    """
    ceil_cell_divisions = ceil(cell_divisions)
    simple_proliferator(treenode, ceil_cell_divisions, 1 / ceil_cell_divisions)
    num_descendants = floor(2**cell_divisions)
    num_leaves_to_remove = 2**ceil_cell_divisions - num_descendants
    every_other_leaf = treenode.get_leaves()[::2]
    for leaf in rng.choice(every_other_leaf, size=num_leaves_to_remove, replace=False):
        parent = leaf.up
        child_dist = leaf.dist
        leaf.delete(preserve_branch_length=False, prevent_nondicotomic=False)
        # extend branch length of nodes that have become leaves:
        if len(parent.children) == 0:
            parent.dist = parent.dist + child_dist
