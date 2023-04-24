r"""Utility functions ^^^^^^^^^^^^^^^^^"""

import ete3
from gcdyn.bdms import TreeNode, TreeError
import numpy as np
from collections import defaultdict
from scipy.stats import uniform


def simple_fivemer_contexts(sequence: str):
    r"""Decompose a sequence into a list of its 5mer contexts.

    Args:
        sequence: A nucleotide sequence.
    """
    return [sequence[(i - 2) : (i + 3)] for i in range(2, len(sequence) - 2)]


def padded_fivemer_contexts_of_paired_sequences(sequence: str, chain_2_start_idx: int):
    r"""Given a pair of sequences with two chains, split them apart at the given
    index, then generate all of the 5mer contexts for each of the subsequences,
    padding appropriately.

    Args:
        sequence: A nucleotide sequence.
        chain_2_start_idx: The index at which to split the sequence into two
    """
    chain_1_seq = "NN" + sequence[:chain_2_start_idx] + "NN"
    chain_2_seq = "NN" + sequence[chain_2_start_idx:] + "NN"
    return simple_fivemer_contexts(chain_1_seq) + simple_fivemer_contexts(chain_2_seq)


def write_leaf_sequences_to_fasta(
    tree: ete3.TreeNode, file_path: str, naive: bool = None
):
    r"""Write the sequences at the leaves of a tree to a FASTA file, potentially
    including a naive sequence.

    Args:
        tree: A tree with sequences at the leaves.
        file_path: The path to the FASTA file to write.
        naive: Flag to include a naive root sequence in output.
    """
    sequence_dict = {leaf.name: leaf.sequence for leaf in tree.iter_leaves()}
    if naive is not None:
        sequence_dict["naive"] = naive
    with open(file_path, "w") as fp:
        for name, sequence in sequence_dict.items():
            fp.write(f">{name}\n")
            fp.write(f"{sequence}\n")


def sample_trees(
    n,
    init_x=0,
    seed=None,
    print_info=True,
    extant_sampling_probability=1,
    **evolve_kwargs,
):
    r"""Returns a sequence of n simulated trees.

    Args:
        n: Number of trees to evolve.
        init_x: Phenotype of the root node of each tree.
        seed: A seed to initialize the random number generation.
              If ``None``, then fresh, unpredictable entropy will be pulled from the OS.
              If an ``int``, then it will be used to derive the initial state.
              If a :py:class:`numpy.random.Generator`, then it will be used directly.
        print_info: Whether to print a summary statistic of the tree sizes.
        extant_sampling_probability: to be passed to :py:meth:`TreeNode.sample_survivors`
                                     as argument `p`.
        evolve_kwargs: Keyword arguments passed to :py:meth:`TreeNode.evolve`.
    """

    rng = np.random.default_rng(seed)

    trees = []

    encountered_errors = defaultdict(int)

    while len(trees) != n:
        try:
            tree = TreeNode(x=init_x)
            tree.evolve(seed=rng, **evolve_kwargs)
            tree.sample_survivors(p=extant_sampling_probability, seed=seed)
            trees.append(tree)
        except TreeError as err:  # not enough survivors
            encountered_errors[str(err)] += 1
            continue

    if print_info:
        if encountered_errors:
            for error, count in encountered_errors.items():
                print("Notice: obtained error", error, count, "times.")

        print(
            "Success: average of",
            sum(len(list(tree.traverse())) for tree in trees) / len(trees),
            "nodes per tree, over",
            len(trees),
            "trees.",
        )

    return tuple(trees)


def random_transition_matrix(length):
    mat = np.abs(np.array([uniform.rvs(size=length) for _ in range(length)]))

    for i in range(length):
        mat[i, i] = 0
        mat[i, :] /= sum(mat[i, :])

    return mat
