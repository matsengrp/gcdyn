import numpy as np
import copy
import csv
import dill
import os
import sys

from gcdyn.bdms import TreeNode
from gcdyn import utils

# fmt: off

# ----------------------------------------------------------------------------------------
def encode_tree(intree, max_leaf_count=None, ladderize=True, dont_scale=False, mtype='tree', bresp=None):
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

    mtype: default 'tree' returns normal encoded tree. If set to 'fitness', returns two-row
    matrix of fitness values.
    """
    # See the pytest for this method in `tests/test_deep_learning.py`
    # ----------------------------------------------------------------------------------------
    def traverse_inorder(tmptr):
        num_children = len(tmptr.children)
        allowed_children = [1, 2] if tmptr.is_root() else [0, 2]  # root used to have 1 child, so may as well not crash if we get one of those old trees
        if num_children not in allowed_children:
            raise Exception("found %s node with %d children (must have %s)" % ('root' if tmptr.up is None else ('leaf' if tmptr.is_leaf() else 'internal'), num_children, allowed_children))
        for child in tmptr.children[:num_children // 2]:  # trivial loop over single lefthand subtree/node
            yield from traverse_inorder(child)
        yield tmptr
        for child in tmptr.children[num_children // 2:]:  # trivial loop over single rightand subtree/node
            yield from traverse_inorder(child)
    # ----------------------------------------------------------------------------------------
    def fill_leaf(node, leaf_index, previous_ancestor):
        if mtype == 'tree':
            matrix[0, leaf_index] = node.t - previous_ancestor.t
            matrix[2, leaf_index] = node.x
        elif mtype == 'fitness':
            matrix[0, leaf_index] = bresp.λ_phenotype(node.x)
    # ----------------------------------------------------------------------------------------
    def fill_ancestor(node, ancestor_index):
        if mtype == 'tree':
            matrix[1, ancestor_index] = node.t
            matrix[3, ancestor_index] = node.x
        elif mtype == 'fitness':
            matrix[1, ancestor_index] = bresp.λ_phenotype(node.x)
    # ----------------------------------------------------------------------------------------
    assert mtype in ['tree', 'fitness']
    if not dont_scale:
        _, intree = scale_tree(intree)
    # assert utils.isclose(np.mean([lf.t for lf in intree.iter_leaves()]), 1), "trees must be scaled to 1 before encoding"
    if ladderize:
        worktree = intree.copy()  # make a copy so the ladderization doesn't modify the input tree
        utils.ladderize_tree(worktree)
    else:
        worktree = intree

    if max_leaf_count is None:
        max_leaf_count = len(worktree.get_leaves())
    assert len(worktree.get_leaves()) <= max_leaf_count
    matrix = np.zeros((4 if mtype=='tree' else 2, max_leaf_count))

    leaf_index, ancestor_index = 0, 0
    previous_ancestor = worktree  # the root
    for node in traverse_inorder(worktree):
        if node.is_leaf():
            fill_leaf(node, leaf_index, previous_ancestor)
            leaf_index += 1
        else:
            fill_ancestor(node, ancestor_index)
            ancestor_index += 1
            previous_ancestor = node

    return matrix

# ----------------------------------------------------------------------------------------
# NOTE could also make version of this that used original (unencoded) tree to attach nodes/names to each dict
def decode_tree(enc_tree, enc_fit=None): #, max_leaf_count=None, ladderize=True):
    """ Reverse the encoding of a tree, i.e. convert encoded tree <enc_tree> into list of dicts, where each dict has
    info for one node. If an encoded fitness matrix <enc_fit> is set, this also includes fitness information. NOTE
    that this obviously is entirely dependent on the hard coded rows in the encoding function. """
    ndicts = []
    assert len(enc_tree) == 4  # 4 rows in encoded tree (leaf dists, internal dists, leaf phenotypes, internal phenotypes)
    assert len(set(len(r) for r in enc_tree)) == 1  # all rows the same length
    if enc_fit is not None:
        assert len(enc_fit) == 2  # 2 rows in encoded fitnesses (leaf fitnesses, internal fitnesses)
        assert all(len(r)==len(enc_tree[0]) for r in enc_fit)  # all rows the same length as encoded tree rows
    for icol in range(len(enc_tree[0])):
        lfo = {'dist' : enc_tree[0][icol], 'phenotype' : enc_tree[2][icol]}  # leaf node
        nfo = {'dist' : enc_tree[1][icol], 'phenotype' : enc_tree[3][icol]}  # internal node
        if enc_fit is not None:
            lfo['fitness'] = enc_fit[0][icol]
            nfo['fitness'] = enc_fit[1][icol]
        ndicts.extend([lfo, nfo])
    return ndicts

# ----------------------------------------------------------------------------------------
def scale_tree(intree, new_mean_depth=1):
    """Return new tree scaled to average leaf depth <new_mean_depth>.
    Also returns original average branch depth <brlen>"""
    mean_brlen = np.mean([lf.t for lf in intree.iter_leaves()])
    outtree = copy.copy(intree)
    for node in intree.iter_descendants(strategy="preorder"):  # preorder so we don't need a separate loop for .dist
        node.t *= new_mean_depth / mean_brlen
        node.dist = node.t - node.up.t
    assert utils.isclose(np.mean([lf.t for lf in outtree.iter_leaves()]), new_mean_depth)
    return mean_brlen, outtree

# ----------------------------------------------------------------------------------------
def scale_trees(intrees):
    scale_vals = []
    for intr in intrees:
        brlen, sctree = scale_tree(intr)
        scale_vals.append(brlen)
    return scale_vals

# ----------------------------------------------------------------------------------------
def encode_trees(intrees, max_leaf_count=None, ladderize=True, mtype='tree', birth_responses=None):
    """Scale and then encode each tree in intrees, returning list of scale vals (average branch length
    for each tree) and list of scaled, encoded trees."""
    scale_vals, enc_trees = [], []
    for itr, intree in enumerate(intrees):
        brlen, sctree = scale_tree(intree)  # rescale separately so we can store the branch len
        scale_vals.append(brlen)
        enc_trees.append(
            encode_tree(
                sctree,
                max_leaf_count=max_leaf_count,
                ladderize=ladderize,
                dont_scale=True,
                mtype=mtype,
                bresp=None if birth_responses is None else birth_responses[itr],
            )
        )
    return scale_vals, enc_trees

# ----------------------------------------------------------------------------------------
def trivialize_encodings(encoded_trees, model_type, predict_vals, noise=False, max_print=10, n_debug=0, debug=False):
    """Convert encoded_trees (modify in place) to a "trivialized" encoding, i.e. one that replaces the tree
    information with the response parameter values that we're trying to predict.
    sigmoid: replace 4-row encoded tree matrix entries with repeating columns of sigmoid parameters
      (first column is xscale, second is xshift, etc.)
    per-cell: replace both the distance and affinity entry for each node with its fitness (i.e. first and third
      [and second and fourth] rows are now equal)
    """
    # ----------------------------------------------------------------------------------------
    def getval(itree, irow, icol):
        if model_type == 'sigmoid':
            param_vals = predict_vals  # each entry (itree) in predict_vals is a list of three values (xscale, xshift, yscale)
            rval = param_vals[itree][icol % len(param_vals[itree])]
        elif model_type == 'per-cell':
            encoded_fitnesses = predict_vals  # each entry (itree) is a 2 x max_leaf_count matrix with the fitness for each node
            rval = encoded_fitnesses[itree][irow % 2][icol]
        else:
            assert False
        if noise:
            rval += np.random.uniform(-0.1 * rval, 0.1 * rval)
        return rval
    # ----------------------------------------------------------------------------------------
    def estr(tstr, etree):
        def vstrs(le): return ["%5.2f"%v for v in list(le[:max_print])]
        return "    %6s: %s" % (tstr, "\n            ".join(" ".join(vstrs(le)) + " ..." for le in etree))
    # ----------------------------------------------------------------------------------------
    if debug or n_debug > 0:
        print(" trivializing encodings")
        np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))
    for itree, etree in enumerate(encoded_trees):
        if debug or itree < n_debug:
            print("  itree %d" % itree)
            print(estr("before", etree))
        for irow in range(len(etree)):
            for icol in range(len(etree[irow])):
                etree[irow][icol] = getval(itree, irow, icol)
        if debug or itree < n_debug:
            print(estr("after", etree))

# ----------------------------------------------------------------------------------------
def pad_trees(
    trees: list[np.ndarray], min_n_max_leaves: int = 50, debug: bool = False
):
    debug = True
    """Pad a list of encoded trees with zeros so that they're all the same length.
    Returns a new array with the padded trees (does not modify input trees).
    """
    n_leaf_list = [len(t[0]) for t in trees]
    max_leaf_count = max(min_n_max_leaves, max(n_leaf_list))  # model complains if this is 70, i'm not sure why but whatever
    if debug:
        print("    padding encoded trees to max leaf count %d (all leaf counts: %s)" % (max_leaf_count, " ".join(str(c) for c in set(n_leaf_list))))
    padded_trees = []
    for itree, etree in enumerate(trees):
        assert len(set(len(r) for r in etree)) == 1  # and that every row is the same length
        padded_trees.append(
            np.pad(etree, ((0, 0), (0, max_leaf_count - len(etree[0]))))
        )
        if debug and itree == 0:
            before_len = len(etree[0])
            np.set_printoptions(precision=3, suppress=True, linewidth=99999)
            print("  padded length from %d to %d" % (before_len, len(etree[0])))
            # print(etree)
    return np.array(padded_trees)

# ----------------------------------------------------------------------------------------
def write_trees(filename, trees):
    np.save(filename, pad_trees(trees))  # maybe should at some point use savez_compressed()? but size isn't an issue atm (have to pad here since np.save() requires arrays of same dimension

# ----------------------------------------------------------------------------------------
def read_trees(filename: str):
    return np.load(filename)

# ----------------------------------------------------------------------------------------
final_ofn_strs = ["seqs", "trees", "meta", "encoded-trees", "encoded-fitnesses", "responses", "summary-stats"]
model_state_ofn_strs = ["model", "train-scaler", "example-responses"]
sstat_fieldnames = ["tree", "mean_branch_length", "total_branch_length", "carry_cap", "time_to_sampling"]
leaf_meta_fields = ["tree-index", "name", "affinity", "n_muts", "n_muts_aa", "gc", "is_leaf"]

# ----------------------------------------------------------------------------------------
def output_fn(odir, ftype, itrial):
    """Return file name for simulation files of various types."""
    assert ftype in final_ofn_strs + model_state_ofn_strs + [None]
    if itrial is None:
        suffixes = {
            # final ofn:
            "seqs": "fasta",
            "trees": "nwk",
            "encoded-trees": "npy",
            "encoded-fitnesses": "npy",
            "responses": "pkl",
            "meta": "csv",
            "summary-stats": "csv",
            # model state:
            "model": "keras",
            "train-scaler": "pkl",
            "example-responses": "pkl",
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
        writer = csv.DictWriter(mfile, leaf_meta_fields)
        writer.writeheader()
        for lmfo in lmetafos:
            writer.writerow(lmfo)

# ----------------------------------------------------------------------------------------
def write_training_files(outdir, encoded_trees, responses, sstats, encoded_fitnesses=None, dbgstr=""):
    """Write encoded tree .npy, response fcn .pkl, and summary stat .csv files for training/testing on simulation."""
    if dbgstr != "":
        print("      writing %s files to %s" % (dbgstr, outdir))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if encoded_trees is not None:
        write_trees(output_fn(outdir, "encoded-trees", None), encoded_trees)
    if encoded_fitnesses is not None:
        write_trees(output_fn(outdir, "encoded-fitnesses", None), encoded_fitnesses)
    with open(output_fn(outdir, "responses", None), "wb") as pfile:
        dill.dump(responses, pfile)
    write_sstats(output_fn(outdir, "summary-stats", None), sstats)

# fmt: on
