import jax.numpy as np
from gcdyn.tree import Tree
from gcdyn.parameters import Parameters
import unittest


class TestTree(unittest.TestCase):
    def setUp(self):
        T = 3
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

    def test_prune(self):
        original_sampled = set(
            [node for node in self.tree.tree.traverse() if node.event == "sampled"]
        )
        self.tree.prune()

        assert all(
            node.event == "sampled"
            for node in self.tree.tree.traverse()
            if node.is_leaf()
        )
        assert all(
            len(node.children) == 2
            for node in self.tree.tree.traverse()
            if node.event == "birth"
        )
        assert (
            set([node for node in self.tree.tree.traverse() if node.event == "sampled"])
            == original_sampled
        )


if __name__ == "__main__":
    unittest.main()
