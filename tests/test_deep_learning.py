import unittest
from random import shuffle

import ete3
import numpy as np

from gcdyn import utils
from gcdyn import encode


class TestDeepLearning(unittest.TestCase):
    def test_CBLV_encoding(self):
        """
        A test to make sure we can correctly encode the tree
        in Figure 2a of https://doi.org/10.1038/s41467-022-31511-0
        """

        tree = ete3.Tree(
            newick="(a:3, ((b:2, c:1)C:2, (d:1, e:4)D:1)B:1)A:0;", format=3
        )

        # ete3.Tree doesn't have node.t but I rely on that
        tree.t = 0
        tree.x = 0
        for node in tree.iter_descendants("levelorder"):
            node.t = node.up.t + node.dist
            node.x = node.up.x + 1

        encoded_tree = encode.encode_tree(tree, max_leaf_count=10)

        solution = np.array(
            [
                [6, 1, 4, 1, 3, 0, 0, 0, 0, 0],
                [2, 1, 3, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )

        self.assertTrue(np.all(solution == encoded_tree[:2, :]))

    def test_encoding_invariance(self):
        """Ensures the ladderization+encoding process is invariant under tree isomorphism."""

        tree = utils.sample_trees(n=1, t=3, seed=10)[0]
        tree.prune()
        tree.remove_mutation_events()

        tree2 = tree.copy()
        for node in tree2.traverse("postorder"):
            shuffle(node.children)

        encoded_tree = encode.encode_tree(tree)
        encoded_tree2 = encode.encode_tree(tree2)
        self.assertTrue(np.all(encoded_tree == encoded_tree2))

    def test_read_write(self, test_dir='/tmp'):
        """Make sure that tree encoding doesn't change on reading or writing."""

        init_trees = []
        for ttr in utils.sample_trees(n=3, t=3, seed=10):
            ttr.prune()
            ttr.remove_mutation_events()
            init_trees.append(encode.encode_tree(ttr))
        init_trees = encode.pad_trees(init_trees)  # have to pad initial trees since they get padded before writing

        encode.write_trees('%s/test-tree.npy' % test_dir, init_trees)
        read_trees = encode.read_trees('%s/test-tree.npy' % test_dir)

        self.assertTrue(all(np.all(rt == it) for it, rt in zip(init_trees, read_trees)))


if __name__ == "__main__":
    unittest.main()
