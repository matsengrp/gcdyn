import numpy as np
import copy
import csv
import dill
import os

from gcdyn.bdms import TreeNode
from gcdyn import utils

def encode_tree(
    intree: TreeNode,
    max_leaf_count: int = None,
    ladderize: bool = True,
    dont_scale: bool = False,
) -> np.ndarray[float]:
    """
    Returns the "Compact Bijective Ladderized Vector" form of the given
    tree. Does not modify input tree.

    The CBLV has been adapted to include the `x` attribute of every node.
    Thus, in reference to figure 2a (v) in Voznica et. al (2022), two additional
    rows have been appended: a third row of `x` for the nodes in row 1, and a
    fourth row of `x` for the nodes in row 2.

    Note that by default this ladderizes the tree first, so if you've already done
    this you should set ladderize to False.

    If max_leaf_count is not set, it defaults to the number of leaves in the tree.
    It often makes more sense to pad encoded trees to the same size right before passing
    into the model (using encode.pad_trees()) than to try to guess a max_leaf_count
    when initially encoding them.
    """

    # See the pytest for this method in `tests/test_deep_learning.py`

    def traverse_inorder(tmptr):
        num_children = len(tmptr.children)
        assert tmptr.up is None or num_children in {
            0,
            2,
        }, (
            "Only full binary trees are supported, but found node with %d children"
            % num_children
        )

        for child in tmptr.children[
            : num_children // 2
        ]:  # trivial loop over single lefthand subtree/node
            yield from traverse_inorder(child)

        yield tmptr

        for child in tmptr.children[
            num_children // 2 :
        ]:  # trivial loop over single rightand subtree/node
            yield from traverse_inorder(child)

    if not dont_scale:
        _, intree = scale_tree(intree)

    # assert utils.isclose(np.mean([lf.t for lf in intree.iter_leaves()]), 1), "trees must be scaled to 1 before encoding"

    if ladderize:
        worktree = (
            intree.copy()
        )  # make a copy so the ladderization doesn't modify the input tree
        utils.ladderize_tree(worktree)
    else:
        worktree = intree

    if max_leaf_count is None:
        max_leaf_count = len(worktree.get_leaves())
    assert len(worktree.get_leaves()) <= max_leaf_count
    matrix = np.zeros((4, max_leaf_count))

    leaf_index = 0
    ancestor_index = 0
    previous_ancestor = worktree  # the root

    for node in traverse_inorder(worktree):
        if node.is_leaf():
            matrix[0, leaf_index] = node.t - previous_ancestor.t
            matrix[2, leaf_index] = node.x
            leaf_index += 1
        else:
            matrix[1, ancestor_index] = node.t
            matrix[3, ancestor_index] = node.x
            ancestor_index += 1
            previous_ancestor = node

    return matrix


def scale_tree(
    intree: TreeNode,
) -> (float, TreeNode):
    """Scale intree to average branch length 1, i.e. divide all branches by the average branch length.
    Returns initial average branch length brlen and scaled tree outtree."""
    mean_brlen = np.mean([lf.t for lf in intree.iter_leaves()])
    outtree = copy.copy(intree)
    for node in outtree.iter_descendants():
        node.t /= mean_brlen
    assert utils.isclose(np.mean([lf.t for lf in outtree.iter_leaves()]), 1)
    return mean_brlen, outtree


def encode_trees(
    intrees: list[TreeNode],
    max_leaf_count: int = None,
    ladderize: bool = True,
) -> (list[float], list[np.ndarray[float]]):
    """Scale and then encode each tree in intrees, returning list of scale vals (average branch length
    for each tree) and list of scaled, encoded trees."""
    scale_vals, enc_trees = [], []
    for intr in intrees:
        brlen, sctree = scale_tree(
            intr
        )  # rescale separately so we can store the branch len
        scale_vals.append(brlen)
        enc_trees.append(
            encode_tree(
                sctree,
                max_leaf_count=max_leaf_count,
                ladderize=ladderize,
                dont_scale=True,
            )
        )
    return scale_vals, enc_trees


def trivialize_encodings(
    encoded_trees, param_vals, noise=False, max_print=10, n_debug=0, debug=False
):
    """Convert encoded_trees to a "trivialized" encoding, i.e. one that replaces the actual tree
    information with the response parameter values that we're trying to predict."""

    def getval(itree, icol):
        rval = param_vals[itree][icol % len(param_vals[itree])]
        if noise:
            rval += np.random.uniform(-0.1 * rval, 0.1 * rval)
        return rval

    def estr(tstr, entr):
        return "    %6s: %s" % (
            tstr,
            "\n            ".join(
                " ".join("%5.2f" % v for v in list(le[:max_print])) + " ..."
                for le in entr
            ),
        )

    if debug or n_debug > 0:
        print(" trivializing encodings")
        np.set_printoptions(
            edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
        )
    for itree, entr in enumerate(encoded_trees):
        if debug or itree < n_debug:
            print("  itree %d" % itree)
            print(estr("before", entr))
        for irow in range(len(entr)):
            for icol in range(len(entr[irow])):
                entr[irow][icol] = getval(itree, icol)
        if debug or itree < n_debug:
            print(estr("after", entr))


def pad_trees(
    trees: list[np.ndarray], min_n_max_leaves: int = 100, debug: bool = False
):
    """Pad a list of encoded trees with zeros so that they're all the same length.
    Returns a new array with the padded trees (does not modify input trees).
    """
    n_leaf_list = [len(t[0]) for t in trees]
    max_leaf_count = max(
        min_n_max_leaves, max(n_leaf_list)
    )  # model complains if this is 70, i'm not sure why but whatever
    if debug:
        print(
            "    padding encoded trees to max leaf count %d (all leaf counts: %s)"
            % (max_leaf_count, " ".join(str(c) for c in set(n_leaf_list)))
        )
    padded_trees = []
    for itree, etree in enumerate(trees):
        assert len(etree) == 4  # make sure there's 4 rows
        assert (
            len(set(len(r) for r in etree)) == 1
        )  # and that every row is the same length
        padded_trees.append(
            np.pad(etree, ((0, 0), (0, max_leaf_count - len(etree[0]))))
        )
        if debug and itree == 0:
            before_len = len(etree[0])
            np.set_printoptions(precision=3, suppress=True, linewidth=99999)
            print("  padded length from %d to %d" % (before_len, len(etree[0])))
            print(etree)
    return np.array(padded_trees)


def write_trees(
    filename: str,
    trees: list[np.ndarray],
):
    np.save(
        filename, pad_trees(trees)
    )  # maybe should at some point use savez_compressed()? but size isn't an issue atm (have to pad here since np.save() requires arrays of same dimension


def read_trees(filename: str):
    return np.load(filename)

# ----------------------------------------------------------------------------------------
final_ofn_strs = ["seqs", "trees", "leaf-meta", "encoded-trees", "responses", "summary-stats"]
sstat_fieldnames = ["tree", "mean_branch_length", "total_branch_length", "carry_cap", "time_to_sampling"]

# ----------------------------------------------------------------------------------------
def output_fn(odir, ftype, itrial):
    """Return file name for simulation files of various types."""
    assert ftype in final_ofn_strs + [None]
    if itrial is None:
        suffixes = {
            "seqs": "fasta",
            "trees": "nwk",
            "encoded-trees": "npy",
            "responses": "pkl",
            "leaf-meta": "csv",
            "summary-stats": "csv",
        }
        sfx = suffixes.get(ftype, "simu")
    else:
        assert ftype is None
        ftype = "tree_%d" % itrial
        sfx = "pkl"
    return f"{odir}/{ftype}.{sfx}"

# ----------------------------------------------------------------------------------------
def write_sstats(ofn, sstats):
    with open(ofn, "w") as sfile:
        writer = csv.DictWriter(sfile, sstat_fieldnames)
        writer.writeheader()
        for sline in sstats:
            writer.writerow({k: v for k, v in sline.items() if k in sstat_fieldnames})

# ----------------------------------------------------------------------------------------
def write_leaf_meta(ofn, lmetafos):
    with open(ofn, "w") as mfile:
        writer = csv.DictWriter(mfile, ["name", "affinity", "n_muts"])
        writer.writeheader()
        for lmfo in lmetafos:
            writer.writerow(lmfo)

# ----------------------------------------------------------------------------------------
def write_training_files(outdir, encoded_trees, responses, sstats, dbgstr=""):
    """Write encoded tree .npy, response fcn .pkl, and summary stat .csv files for training/testing on simulation."""
    if dbgstr != "":
        print("      writing %s files to %s" % (dbgstr, outdir))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    write_trees(output_fn(outdir, "encoded-trees", None), encoded_trees)
    with open(output_fn(outdir, "responses", None), "wb") as pfile:
        dill.dump(responses, pfile)
    write_sstats(output_fn(outdir, "summary-stats", None), sstats)
