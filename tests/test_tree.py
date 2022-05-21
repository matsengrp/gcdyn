import numpy as np
from gcdyn.tree import Tree
from gcdyn.parameters import Parameters
import unittest


class TestTree(unittest.TestCase):
    def setUp(self):
        T = 4
        seed = 0

        # response function parameters
        θ = np.array([3, 1, 0], dtype=float)
        # death rate
        μ = 1
        # mutation rate
        m = 1
        # sampling efficiency
        ρ = 0.5

        params = Parameters(θ, μ, m, ρ)

        self.tree = Tree(T, seed, params)
        self.assertTrue(len(self.tree.tree) > 10)

    def test_prune(self):
        original_sampled = set(
            [node for node in self.tree.tree.traverse() if node.event == "sampled"]
        )
        self.tree.prune()
        self.assertTrue(all(node.event == "sampled" for node in self.tree.tree))
        self.assertTrue(
            all(
                len(node.children) == 2
                for node in self.tree.tree.traverse()
                if node.event == "birth"
            )
        )
        self.assertEqual(
            set(
                [node for node in self.tree.tree.traverse() if node.event == "sampled"]
            ),
            original_sampled,
        )
        mean_root_to_tip_distance = np.mean(
            np.array([self.tree.tree.get_distance(leaf) for leaf in self.tree.tree])
        )
        for leaf in self.tree.tree:
            self.assertAlmostEqual(
                self.tree.tree.get_distance(leaf), mean_root_to_tip_distance, places=5
            )


if __name__ == "__main__":
    unittest.main()
