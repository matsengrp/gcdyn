import unittest
import numpy as np
from numpy.random import default_rng

from gcdyn.cycles import GC, binary_proliferator, Selector, UniformMutator

rng = default_rng()


class NeutralSelector(Selector):
    def select(self, sequence_list):
        return [(0.9,)] * len(sequence_list)


class TestGC(unittest.TestCase):
    def setUp(self):
        self.T = 5
        self.N0 = 10
        sequence = "AAAA"
        neutral_selector = NeutralSelector()
        uniform_mutator = UniformMutator()

        self.gc = GC(
            sequence,
            binary_proliferator,
            uniform_mutator,
            neutral_selector,
            N0=self.N0,
        )
        self.gc.simulate(self.T, max_tries=np.inf)

        print(self.gc.tree)

    def test_ultrametric(self):
        for leaf in self.gc.tree:
            self.assertTrue(self.gc.tree.get_distance(leaf) == self.T)

    def test_root_branches(self):
        if self.N0 > 1:
            for child in self.gc.tree.children:
                self.assertTrue(child.dist == 0)

    def test_pruned(self):
        for node in self.gc.tree.traverse():
            self.assertTrue(not node.terminated)


if __name__ == "__main__":
    unittest.main()
