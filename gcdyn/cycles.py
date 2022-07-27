r"""Germinal center light zone / dark-zone cycles simulator.

Borrowing from `gctree <https://github.com/matsengrp/gctree/blob/master/gctree/mutation_model.py>`_
"""

from typing import Callable, List, Optional, Tuple
from ete3 import TreeNode
import numpy as np
from numpy.random import default_rng
from gcdyn.fitness import Fitness
from gcdyn.phenotype import DMSPhenotype
from math import floor, ceil, isclose
from abc import ABC
import pandas as pd
from math import exp


class ExtinctionError(Exception):
    """The simulation has resulted in extinction of all lineages."""


class Mutator(ABC):
    r"""A class for GC mutators, which mutate a sequence over some period of time."""

    def mutate(self, time: float, rng: np.random.Generator) -> str:
        pass


class UniformMutator(Mutator):
    def mutate(
        self, sequence: str, time: float, rng: np.random.Generator = default_rng()
    ) -> str:
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


class FivemerMutator(Mutator):
    def __init__(
        self,
        mutability_csv: str,
        substitution_csv: str,
        igk_idx: int = 336,
    ):
        """AID hotspot-aware mutation model using mutability values at each
        nucleotide 5mer and substitution probabilities.

        Args:
            mutability_csv: path to CSV with rows representing 5mers and mutability value
            substitution_csv: path to CSV with rows representing 5mers and probability of substitution to each nucleotide in columns
            igk_idx: index of Ig light chain starting position

        Returns:
            sequence: sequence with mutations based on given mutabilities and substitutions
        """

        self.mutability = pd.read_csv(mutability_csv, sep=" ", index_col=0).squeeze(
            "columns"
        )
        self.substitution = pd.read_csv(substitution_csv, sep=" ", index_col=0)
        self.igk_idx = igk_idx

    def mutate(
        self, sequence: str, time: float, rng: np.random.Generator = default_rng()
    ) -> str:
        """AID hotspot-aware mutation model using mutability values at each
        nucleotide 5mer and substitution probabilities.

        Args:
            sequence: initial sequence, consisting of characters ``ACGT``
            time: exposure time
            rng:  random number generator

        Returns:
            sequence: sequence with mutations based on given mutabilities and substitutions
        """

        sequence_H = "NN" + sequence[: self.igk_idx] + "NN"
        sequence_K = "NN" + sequence[self.igk_idx :] + "NN"
        # mutabilities of each nucleotide
        contexts = [
            sequence_H[(i - 2) : (i + 3)] for i in range(2, len(sequence_H) - 2)
        ] + [sequence_K[(i - 2) : (i + 3)] for i in range(2, len(sequence_K) - 2)]
        mutabilities = np.array([self.mutability[context] for context in contexts])
        t = 0
        while True:
            t += rng.exponential()
            if t < time:
                i = rng.choice(len(mutabilities), p=mutabilities / sum(mutabilities))
                sub_nt = rng.choice(
                    self.substitution.columns,
                    p=self.substitution.loc[contexts[i]].fillna(0),
                )
                sequence = sequence[:i] + sub_nt + sequence[(i + 1) :]
                # update contexts and mutabilities
                for j in range(i - 2, i + 3):
                    if 0 <= j < len(sequence):
                        contexts[j] = (
                            contexts[j][: 2 + (i - j)]
                            + sub_nt
                            + contexts[j][3 + (i - j) :]
                        )
                        mutabilities[j] = self.mutability[contexts[j]]
            else:
                break
        return sequence


class Selector(ABC):
    r"""A class for GC selectors, which determine the fitness for a list of nucleotide sequences.
    Selectors return a list of tuples that can be interpreted by a proliferator method"""

    def select(self, sequence_list: List[str], competition: bool = True) -> List[Tuple]:
        """Assigns the fitness for each sequence, with normalization depending
        on whether competition is considered.

        Args:
            sequence_list: list of nucleotide sequences
            competition: presence of competition in the light zone

        Returns:
            cell_tuples: list of tuples to be interpreted by a proliferator method
        """
        pass


class UniformSelector(Selector):
    """Uniform selector assigning fitness of each sequence to an equal
    value."""

    def __init__(self, fitness_value: float = None):
        """
        Args:
            fitness_value: fitness value for each sequence if normalization (``competition``) is off
        """
        if fitness_value is not None:
            self.fitness_value = fitness_value

    def select(
        self, sequence_list: List[str], competition: bool = True
    ) -> List[Tuple[float]]:
        """Uniform selector assigning fitness of each sequence to an equal
        value.

        Args:
            sequence_list: list of sequences for fitness assignment
            competition: if ``True``, competition for T cell help in the light zone is expected,
            and the fitness is set to 1/`len(sequence_list)`. Otherwise, fitness is set to `fitness_value`

        Returns:
            fitness_values: list of tuples containing the assigned number of cell divisions (equal values)
        """
        if competition:
            cell_tuples = [
                tuple([1 / len(sequence_list)]) for _ in range(len(sequence_list))
            ]
        else:
            cell_tuples = [
                tuple([self.fitness_value]) for _ in range(len(sequence_list))
            ]
        return cell_tuples


class ThreeStepSelector(Selector):
    """A Selector that uses DMS data as well as a discrete amount of T cell
    help that is assigned."""

    def __init__(
        self,
        igh_frame: int = 1,
        igk_frame: int = 1,
        igk_idx: int = 336,
        naive_sites_path: str = "https://raw.githubusercontent.com/jbloomlab/Ab-CGGnaive_DMS/main/data/CGGnaive_sites.csv",
        model_path: str = "Linear.model",
        tdms_phenotypes: List[str] = ["delta_log10_KD", "delta_expression"],
        log10_naive_KD: float = -10.43,
        concentration_antigen: float = 10 ** (-9),
    ):
        """Initializes values for a DMSPhenotype to calculate KD based on
        sequence from torchDMS model if KDs are not provided, and sets antigen
        concentration for determining antigen bound.

        Args:
            igh_frame: frame for translation of Ig heavy chain
            igk_frame: frame for translation of Ig light chain
            igk_idx: index of Ig light chain starting position
            naive_sites_path: path to CSV lookup table for converting from scFv CDS indexed site numbering to heavy/light chain IMGT numbering
            model_path: path to ``torchdms`` model for heavy and light chain
            tdms_phenotypes: names of phenotype values produced by passed-in ``torchdms`` model (``delta_log10_KD`` expected as a phenotype)
            log10_naive_KD: KD of naive Ig
            concentration_antigen: molar concentration of antigen to determine antigen bound
        """
        self.igh_frame = igh_frame
        self.igk_frame = igk_frame
        self.igk_idx = igk_idx
        self.naive_sites_path = naive_sites_path
        self.model_path = model_path
        self.tdms_phenotypes = tdms_phenotypes
        self.log10_naive_KD = log10_naive_KD
        self.concentration_antigen = concentration_antigen

    def select(
        self,
        sequence_list: List[str],
        competition: bool = True,
        kd_list: List[float] = None,
        total_t_cell_help: int = 50,
    ) -> List[Tuple[float]]:
        """Produce the predicted number of cell divisions for a list of
        sequences with discrete units of T cell help. T cell help is
        distributed semi-randomly, with probabilites based on normalized signal
        from antigen bound.

        Args:
            sequence_list: list of nucleotide sequences
            competition: presence of competition to determine whether signal is normalized
            kd_list: list of KD values to use instead of calculating KD values using DMSPhenotype
            total_t_cell_help: total units of T cell help to be distributed in germinal center

        Returns:
            fitnesses: tuple with number of cell divisions for each sequence
        """
        if kd_list is None:
            phenotype = DMSPhenotype(
                self.igh_frame,
                self.igk_frame,
                self.igk_idx,
                self.naive_sites_path,
                self.model_path,
                self.tdms_phenotypes,
                self.log10_naive_KD,
            )
            kd_list = phenotype.calculate_KD(sequence_list)
        competencies = self.norm1(kd_list)
        norm_signals = self.norm2_sigmoid(competencies, competition=competition)
        selected_seqs = self._norm3_prob_distribution(norm_signals, total_t_cell_help)
        return [tuple([float(selected_seq)]) for selected_seq in selected_seqs]

    def norm1(self, KD_list, antigen_frac_limit: float = 0.2):
        """For each KD value in ``KD_list``, determines the fraction of antigen bound using the Hill equation and
        returns a list containing that value or 0, if the fraction is below ``antigen_frac_limit``.
        Args:
            KD_list: list of KD values to use (with the concentration antigen) to determine the amount of antigen bound.
            antigen_frac_limit: lower limit (inclusive) for the fraction antigen bound to be returned.

        Returns:
            competencies: a list with the competency (fraction antigen bound or 0) for each KD in the ``KD_list``.
        """
        competencies = []
        for KD in KD_list:
            theta = self.concentration_antigen / (KD + self.concentration_antigen)
            if theta < antigen_frac_limit:
                competencies.append(0)
            else:
                competencies.append(theta)
        return competencies

    def norm2_sigmoid(
        self,
        competencies: List[float],
        curve_steepness: float = 10,
        midpoint_competency: float = 0.5,
        competition: bool = True,
    ):
        """Maps the input competencies to a signal between 0 and 1 using a sigmoidal transformation.
        Args:
            competencies: list of competencies between 0 and 1
            curve_steepness: logistic growth rate of signal
            midpoint_competency: value of input competency to set as midpoint
            competition: presence of competition to determine whether signal is normalized

        Returns:
            signals: normalized or unnormalized signals between 0 and 1 based on sigmoidal transformation of competencies.
        """
        unnorm_signals = []
        for competency in competencies:
            if competency == 0:
                unnorm_signals.append(0)
            else:
                unnorm_signals.append(
                    1
                    / (
                        1
                        + exp(-1 * curve_steepness * (competency - midpoint_competency))
                    )
                )
        if not competition:
            return unnorm_signals
        sum_signals = sum(unnorm_signals)
        norm_signals = [signal / sum_signals for signal in unnorm_signals]
        return norm_signals

    def _norm2_quartile(self, competencies: List[float]):
        """Determines signal based on competency based on the distance from the
        quartile value.

        Args:
            competencies: list of competencies between 0 and 1
        Returns:
            signals: signals between 0 and 1.
        """
        quartile_competency = np.quantile(competencies, [0.25])[0]
        signals = []
        for competency in competencies:
            signals.append(max(0, competency - quartile_competency))
        return signals

    def _norm3_prob_distribution(
        self,
        norm_signals,
        total_t_cell_help,
        max_help: int = 2,
        rng: np.random.Generator = default_rng(),
    ):
        """Produces a list with integer T cell help values using
        ``norm_signals`` as the probability of receiving help.

        Args:
            norm_signals: T cell help signal for each sequence (must add to 1).
            total_t_cell_help: total units of T cell help to be distributed
            max_help: maximum amount of help to assign to a single sequence
            rng: random number generator

        Returns:
        """
        indices = np.arange(len(norm_signals))
        amt_help = np.zeros(len(norm_signals), dtype=int)

        # If T cell help available exceeds the number of non-zero signals, every non-zero signal produces ``max_help``.
        if total_t_cell_help >= max_help * np.count_nonzero(norm_signals):
            for i in range(len(norm_signals)):
                if norm_signals[i] > 0:
                    amt_help[i] = max_help
        # Otherwise, sample with weighted probabilities ``norm_signals``, re-sampling to not exceed ``max_help``.
        else:
            for _ in range(total_t_cell_help):
                chosen_index = rng.choice(indices, p=norm_signals)
                while amt_help[chosen_index] > max_help - 1:
                    chosen_index = rng.choice(indices, p=norm_signals)
                amt_help[chosen_index] += 1
        return amt_help


class DMSSelector(Selector):
    r"""A class for a GC selector which determines fitness for nucleotide sequences by using DMS models of
    affinity.
    """

    def __init__(
        self,
        slope: float = 3,
        y_intercept: float = 1,
        igh_frame: int = 1,
        igk_frame: int = 1,
        igk_idx: int = 336,
        naive_sites_path: str = "https://raw.githubusercontent.com/jbloomlab/Ab-CGGnaive_DMS/main/data/CGGnaive_sites.csv",
        model_path: str = "Linear.model",
        tdms_phenotypes: List[str] = ["delta_log10_KD", "delta_expression"],
        log10_naive_KD: float = -10.43,
        fitness_method: Callable[
            [List[float]], List[float]
        ] = Fitness.sigmoidal_fitness,
    ):
        """
        Args:
            slope: coefficient of relationship between normalized T cell help and cell divisions
            y_intercept: cell divisions per cycle with no T cell help
            igh_frame: frame for translation of Ig heavy chain
            igk_frame: frame for translation of Ig light chain
            igk_idx: index of Ig light chain starting position
            naive_sites_path: path to CSV lookup table for converting from scFv CDS indexed site numbering to heavy/light chain IMGT numbering
            model_path: path to ``torchdms`` model for antibody sequences
            tdms_phenotypes: names of phenotype values produced by passed-in ``torchdms`` model (``delta_log10_KD`` expected as a phenotype)
            log10_naive_KD: KD of naive Ig
            fitness_method: method to map from KD to T cell help, taking in a list of :math:`K_D` values and producing a list of absolute T cell help quantities
        """
        self.phenotype = DMSPhenotype(
            igh_frame,
            igk_frame,
            igk_idx,
            naive_sites_path,
            model_path,
            tdms_phenotypes,
            log10_naive_KD,
        )
        self.fitness = Fitness(fitness_method)
        self.slope = slope
        self.y_intercept = y_intercept

    def select(
        self, sequence_list: List[str], competition: bool = True
    ) -> List[Tuple[float]]:
        """Produce the predicted number of cell divisions for a list of
        sequences using a sigmoidal relationship between T cell help and
        fitness.

        Args:
            sequence_list: list of DNA sequences
            competition: if ``True``, competition for T cell help in the light zone is expected, and the cell divisions
            are determined from only the sequence's predicted amount of antigen bound
        Returns:
            cell_divs: a list containing a tuple with the number of cell divisions (non-integer) for each sequence
        """
        sig_fit_df = self.fitness.normalized_fitness_df(
            sequence_list, calculate_KD=self.phenotype.calculate_KD
        )
        if competition:
            t_cell_help_col = "normalized_t_cell_help"
        else:
            t_cell_help_col = "t_cell_help"
        cell_divs = self.fitness.cell_divisions_from_tfh_linear(
            sig_fit_df[t_cell_help_col], self.slope, self.y_intercept
        )
        cell_tuples = [tuple([cell_div]) for cell_div in cell_divs]
        return cell_tuples


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
        mutator: Mutator,
        selector: Selector,
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

    def step(self, enforce_timescale: bool = True, competition: bool = True) -> None:
        r"""Simulate one cycle.

        Args:
            enforce_timescale: if ``True``, the timescale of the DZ proliferation tree must be consistent with the global timescale (one unit per step)
            competition: if ``True``, competition in the LZ is expected (affects Selector output)
        """
        fitnesses = self.selector.select(
            [leaf.sequence for leaf in self.alive_leaves], competition
        )
        for leaf, args in zip(self.alive_leaves, fitnesses):
            self.proliferator(leaf, *args, rng=self.rng)
            for node in leaf.iter_descendants():
                node.sequence = self.mutator.mutate(
                    node.up.sequence, node.dist, rng=self.rng
                )
            if leaf.is_leaf():
                leaf.terminated = True
            if enforce_timescale:
                for subleaf in leaf:
                    if (
                        not subleaf.terminated
                        and subleaf != leaf
                        and not isclose(leaf.get_distance(subleaf), 1)
                    ):
                        raise ValueError(
                            f"DZ subtree timescale is not consistent with simulation time, val = {leaf.get_distance(subleaf)}"
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
        competition: bool = True,
    ) -> None:
        r"""Simulate.

        Args:
            T: number of cycles to simulate
            prune: prune to the tree induced by the surviving lineages
            max_tries: try this many times to simulate a tree that doesn't go extinct
            enforce_timescale: if ``True``, the timescale of the DZ proliferation trees must be consistent with the global timescale
            competition: if ``True``, competition in the LZ is expected (affects Selector output)
        """
        success = False
        n_tries = 0
        while not success:
            try:
                for _ in range(T):
                    self.step(
                        enforce_timescale=enforce_timescale, competition=competition
                    )
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
    treenode: TreeNode, p: float, *args, rng: np.random.Generator = default_rng()
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
    if cell_divisions > 0:
        if dist is None:
            dist = 1 / cell_divisions
        for _ in range(2):
            child = TreeNode()
            child.dist = dist
            child.sequence = treenode.sequence
            child.terminated = False
            treenode.add_child(child)
            simple_proliferator(child, cell_divisions - 1, dist)


def cell_div_balanced_proliferator(
    treenode: TreeNode,
    cell_divisions: float,
    rng: np.random.Generator = default_rng(),
) -> None:
    r"""Populates descendants on a tree node based on the number of cell divisions indicated, producing the number of children expected by :math:`2^{cell\_divisions}`, where ``cell_divisions`` is not necessarily an integer value.
    A full binary tree is generated, then leaf nodes are removed at random from the set of alternating leaf nodes until there are the expected number of children.

    Args:
        treenode: root node to populate from
        cell_divisions: number of cell divisions from the cell represented by ``treenode``
        kd: calculated KD value for sequence
        fitness: calculated fitness for sequence
        rng: random number generator
    """
    if cell_divisions > 0:
        ceil_cell_divisions = ceil(cell_divisions)
        simple_proliferator(
            treenode, ceil_cell_divisions, 1 / max(1, ceil_cell_divisions)
        )
        num_descendants = floor(2**cell_divisions)
        num_leaves_to_remove = 2**ceil_cell_divisions - num_descendants
        every_other_leaf = treenode.get_leaves()[::2]
        for leaf in rng.choice(
            every_other_leaf, size=num_leaves_to_remove, replace=False
        ):
            parent = leaf.up
            child_dist = leaf.dist
            leaf.delete(preserve_branch_length=False, prevent_nondicotomic=False)
            # extend branch length of nodes that have become leaves:
            if len(parent.children) == 0:
                parent.dist = parent.dist + child_dist
