from gcdyn import models, utils
import numpy as np
import ete3
import unittest


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

        utils.ladderize_tree(tree)

        encoded_tree = models.NeuralNetworkModel._encode_tree(tree, max_leaf_count=10)

        solution = np.array(
            [
                [6, 1, 4, 1, 3, 0, 0, 0, 0, 0],
                [2, 1, 3, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )

        self.assertTrue(np.all(solution == encoded_tree[:2, :]))


if __name__ == "__main__":
    unittest.main()
