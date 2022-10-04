import numpy as np
from gcdyn import bdms
import unittest


class TestTreeNode(unittest.TestCase):
    def setUp(self):
        self.tree = bdms.TreeNode()
        for seed in range(1000):
            try:
                self.tree.evolve(
                    5,
                    birth_rate=bdms.SigmoidResponse(1, 0, 2, 0),
                    death_rate=bdms.ConstantResponse(1),
                    mutation_rate=bdms.ConstantResponse(1),
                    mutator=bdms.GaussianMutator(-1, 1),
                    min_survivors=20,
                    seed=seed,
                )
                break
            except bdms.TreeError:
                continue

    def test_sample_survivors(self):
        self.tree.sample_survivors(n=10)
        self.assertTrue(
            all(leaf.event in ("survival", "sampling", "death") for leaf in self.tree)
        )
        self.assertTrue(
            all(
                len(node.children) == 2
                for node in self.tree.traverse()
                if node.event == "birth"
            )
        )
        mean_root_to_tip_distance = np.mean(
            np.array(
                [
                    self.tree.get_distance(leaf)
                    for leaf in self.tree
                    if leaf.event == "sampling"
                ]
            )
        )
        for leaf in self.tree:
            if leaf.event == "sampling":
                self.assertAlmostEqual(
                    self.tree.get_distance(leaf), mean_root_to_tip_distance, places=5
                )

    def test_prune(self):
        self.tree.sample_survivors(n=10)
        original_sampled = set(
            [node for node in self.tree.traverse() if node.event == "sampling"]
        )
        self.tree.prune()
        self.assertTrue(all(leaf.event == "sampling" for leaf in self.tree))
        self.assertTrue(
            all(
                len(node.children) == 2
                for node in self.tree.traverse()
                if node.event == "birth"
            )
        )
        self.assertTrue(
            not any(len(node.children) == 1 for node in self.tree.iter_descendants())
        )
        self.assertEqual(
            set([node for node in self.tree.traverse() if node.event == "sampling"]),
            original_sampled,
        )
        mean_root_to_tip_distance = np.mean(
            np.array([self.tree.get_distance(leaf) for leaf in self.tree])
        )
        for leaf in self.tree:
            self.assertAlmostEqual(
                self.tree.get_distance(leaf), mean_root_to_tip_distance, places=5
            )


if __name__ == "__main__":
    unittest.main()
