r"""Utility functions ^^^^^^^^^^^^^^^^^"""

from collections import defaultdict

import ete3
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform

from gcdyn.bdms import TreeError, TreeNode


def simple_fivemer_contexts(sequence: str):
    r"""Decompose a sequence into a list of its 5mer contexts.

    Args:
        sequence: A nucleotide sequence.
    """
    return tuple([sequence[(i - 2) : (i + 3)] for i in range(2, len(sequence) - 2)])


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

    return sequence_dict


def ladderize_tree(tree, attr="x"):
    """
    Ladderizes the given tree.
    Adapts the procedure described in Voznica et. al (2022) for trees whose leaves
    may occur at the same time.

    *This is done in place!*
    Assumes that the `node.t` time attribute is ascending from root to leaf.

    First, we compute the following values for each node in the tree:
        1. The time of the leaf in the subtree at/below this node which is
           closest to present time
        2. The time of the ancestor node immediately prior to that leaf
        3. The attribute value `attr` (given in arguments) of that leaf

    Then, every node has its child subtrees reordered to sort by these values, decreasing from left to right (which corresponds to most recent and largest `attr` first).
    Values 2 and 3 are tie-breakers for value 1.

    Voznica, J., A. Zhukova, V. Boskova, E. Saulnier, F. Lemoine, M. Moslonka-Lefebvre, and O. Gascuel. “Deep Learning from Phylogenies to Uncover the Epidemiological Dynamics of Outbreaks.” Nature Communications 13, no. 1 (July 6, 2022): 3896. https://doi.org/10.1038/s41467-022-31511-0.
    """

    sort_criteria = defaultdict(list)

    for node in tree.traverse("postorder"):
        if node.is_leaf():
            sort_criteria[node.name] = [node.t, node.up.t, getattr(node, attr)]
        else:
            sort_criteria[node.name] = sorted(
                (sort_criteria[child.name] for child in node.children), reverse=True
            )[0]

    for node in tree.traverse("postorder"):
        if len(node.children) > 1:
            node.children = sorted(
                node.children,
                key=lambda node: sort_criteria[node.name],
                reverse=True,
            )


def sample_trees(
    n,
    init_x=0,
    seed=None,
    print_info=True,
    extant_sampling_probability=1,
    extinct_sampling_probability=1,
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
        extant_sampling_probability: To be passed to :py:meth:`TreeNode.sample_survivors` as argument `p`.
        kwargs: Keyword arguments passed to :py:meth:`TreeNode.evolve`.
    """

    rng = np.random.default_rng(seed)

    trees = []
    encountered_errors = defaultdict(int)

    while len(trees) != n:
        try:
            tree = TreeNode()
            tree.x = init_x
            tree.evolve(seed=rng, **evolve_kwargs)
            tree.sample_survivors(p=extant_sampling_probability, seed=rng)
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


def random_transition_matrix(length, seed=None):
    rng = np.random.default_rng(seed)

    mat = np.abs(
        np.array([uniform.rvs(size=length, random_state=rng) for _ in range(length)])
    )

    for i in range(length):
        mat[i, i] = 0
        mat[i, :] /= sum(mat[i, :])

    return mat


def plot_responses(*responses, x_range=(-10, 10), **named_responses):
    x_array = np.linspace(*x_range)

    plt.figure()

    for response in responses:
        plt.plot(x_array, response.λ_phenotype(x_array), color="black", alpha=0.5)

    for name, response in named_responses.items():
        plt.plot(x_array, response.λ_phenotype(x_array), label=name)

    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.xlim(*x_range)

    if named_responses:
        plt.legend()

    plt.show()


# ----------------------------------------------------------------------------------------
# The following functions (mostly with <arglist> in the name) are for manipulating lists of command line arguments
# (here called "clist", e.g. from sys.argv) to, for instance, allow a script to modify its own arguments for use in running
# subprocesses of itself with similar command lines. They've been copied from partis/utils.py.

# return true if <argstr> is in <clist>
def is_in_arglist(
    clist, argstr
):  # accounts for argparse unique/partial matches (update: we no longer allow partial matches)
    return len(arglist_imatches(clist, argstr)) > 0


# return list of indices matching <argstr> in <clist>
def arglist_imatches(clist, argstr):
    assert (
        argstr[:2] == "--"
    )  # this is necessary since the matching assumes that argparse has ok'd the uniqueness of an abbreviated argument UPDATE now we've disable argparse prefix matching, but whatever
    return [
        i for i, c in enumerate(clist) if argstr == c
    ]  # NOTE do *not* try to go back to matching just the start of the argument, in order to make that work you'd need to have access to the whole list of potential arguments in bin/partis, and you'd probably screw it up anyway


# return index of <argstr> in <clist>
def arglist_index(clist, argstr):
    imatches = arglist_imatches(clist, argstr)
    if len(imatches) == 0:
        raise Exception("'%s' not found in cmd: %s" % (argstr, " ".join(clist)))
    if len(imatches) > 1:
        raise Exception("too many matches")
    return imatches[0]


# replace the argument to <argstr> in <clist> with <replace_with>, or if <argstr> isn't there add it. If we need to add it and <insert_after> is set, add it after <insert_after>
def replace_in_arglist(clist, argstr, replace_with, insert_after=None, has_arg=False):
    if clist.count(None) > 0:
        raise Exception("None type value in clist %s" % clist)
    if not is_in_arglist(clist, argstr):
        if insert_after is None or insert_after not in clist:  # just append it
            clist.append(argstr)
            clist.append(replace_with)
        else:  # insert after the arg <insert_after>
            insert_in_arglist(
                clist, [argstr, replace_with], insert_after, has_arg=has_arg
            )
    else:
        clist[arglist_index(clist, argstr) + 1] = replace_with


# insert list <new_arg_strs> after <argstr> (unless <before> is set),  Use <has_arg> if <argstr> has an argument after which the insertion should occur
def insert_in_arglist(
    clist, new_arg_strs, argstr, has_arg=False, before=False
):  # set <argstr> to None to put it at end (yeah it probably should've been a kwarg)
    i_insert = len(clist)
    if argstr is not None:
        i_insert = clist.index(argstr) + (2 if has_arg else 1)
    clist[i_insert:i_insert] = new_arg_strs


def remove_from_arglist(clist, argstr, has_arg=False):
    if clist.count(None) > 0:
        raise Exception("None type value in clist %s" % clist)
    imatches = arglist_imatches(clist, argstr)
    if len(imatches) == 0:
        return
    if len(imatches) > 1:
        assert False  # not copying this fcn from partis (shouldn't get here atm, but leaving it commented to provide context in case it does get triggered)
        # imatches = reduce_imatches(imatches, clist, argstr)
    iloc = imatches[0]
    # if clist[iloc] != argstr:
    #     print '  %s removing abbreviation \'%s\' from sys.argv rather than \'%s\'' % (color('yellow', 'warning'), clist[iloc], argstr)
    if has_arg:
        clist.pop(iloc + 1)
    clist.pop(iloc)
    return clist  # NOTE usually we don't use the return value (just modify it in memory), but at least in one place we're calling with a just-created list so it's nice to be able to use the return value
