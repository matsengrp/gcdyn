import numpy as np

from gcdyn.bdms import TreeNode
from gcdyn import utils


def encode_tree(
        tree: TreeNode, max_leaf_count: int = None, ladderize: bool = True,
) -> np.ndarray[float]:
    """
    Returns the "Compact Bijective Ladderized Vector" form of the given
    ladderized tree.

    The CBLV has been adapted to include the `x` attribute of every node.
    Thus, in reference to figure 2a (v) in Voznica et. al (2022), two additional
    rows have been appended: a third row of `x` for the nodes in row 1, and a
    fourth row of `x` for the nodes in row 2.

    Note that by default this ladderizes the tree first, so if you've already done
    this you should set ladderizes to False.

    If max_leaf_count is not set, it defaults to the number of leaves in the tree.
    It often makes more sense to pad encoded trees to the same size right before passing
    into the model (using encode.pad_trees()) than to try to guess a max_leaf_count
    when initially encoding them.
    """

    # See the pytest for this method in `tests/test_deep_learning.py`

    def traverse_inorder(tree):
        num_children = len(tree.children)
        assert tree.up is None or num_children in {
            0,
            2,
        }, "Only full binary trees are supported."

        for child in tree.children[: num_children // 2]:
            yield from traverse_inorder(child)

        yield tree

        for child in tree.children[num_children // 2 :]:
            yield from traverse_inorder(child)

    if ladderize:
        utils.ladderize_tree(tree)

    if max_leaf_count is None:
        max_leaf_count = len(tree.get_leaves())
    assert len(tree.get_leaves()) <= max_leaf_count
    matrix = np.zeros((4, max_leaf_count))

    leaf_index = 0
    ancestor_index = 0
    previous_ancestor = tree  # the root

    for node in traverse_inorder(tree):
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


def pad_trees(trees: list[np.ndarray], min_n_max_leaves: int = 100):
    """ Pad a list of encoded trees with zeros so that they're all the same length.
    Returns a new array with the padded trees (does not modify input trees).
    """
    n_leaf_list = ([len(t[0]) for t in trees])
    max_leaf_count = max(min_n_max_leaves, max(n_leaf_list))  # model complains if this is 70, i'm not sure why but whatever
    print('    padding encoded trees to max leaf count %d (all leaf counts: %s)' % (max_leaf_count, ' '.join(str(c) for c in set(n_leaf_list))))
    padded_trees = []
    for itree, etree in enumerate(trees):
        assert len(etree) == 4  # make sure there's 4 rows
        assert len(set(len(r) for r in etree)) == 1  # and that every row is the same length
        padded_trees.append(np.pad(etree, ((0, 0), (0, max_leaf_count - len(etree[0])))))
        # if itree == 0:
        #     before_len = len(etree[0])
        #     np.set_printoptions(precision=3, suppress=True, linewidth=99999)
        #     print('  padded length from %d to %d' % (before_len, len(etree[0])))
        #     print(etree)
    return np.array(padded_trees)


def write_trees(
        filename: str, trees: list[np.ndarray],
        ):
    np.save(filename, pad_trees(trees))  # maybe should use savez_compressed()? size isn't an issue atm tho


def read_trees(
        filename: str
):
    return np.load(filename)

