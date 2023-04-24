import numpy as np
from gcdyn import bdms, mutators, poisson
import unittest


class TestTreeNode(unittest.TestCase):
    def setUp(self):
        self.tree = bdms.TreeNode()
        self.tree.x = 0.0
        for seed in range(1000):
            try:
                self.tree.evolve(
                    5,
                    poisson.SigmoidResponse(1, 0, 2, 0),
                    poisson.ConstantResponse(1),
                    poisson.ConstantResponse(1),
                    mutators.GaussianMutator(-1, 1),
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
        mutation_count = sum(node.event == "mutation" for node in self.tree.traverse())
        self.tree.remove_mutation_events()
        self.assertTrue(
            all(
                len(node.children) == 2
                for node in self.tree.traverse()
                if node.event == "birth"
            )
        )
        mutation_count2 = sum(node.n_mutations for node in self.tree.traverse())
        self.assertEqual(mutation_count, mutation_count2)
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
