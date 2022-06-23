import unittest
from ete3 import TreeNode
from numpy.random import default_rng

from gcdyn.cycles import GC

rng = default_rng()


def subtree_simulator(treenode, p, q):
    r"""binary offspring distribution (Galton-Watson)"""
    if rng.random() > p:
        treenode.terminated = True
    else:
        for _ in range(2):
            child = TreeNode()
            child.dist = 1
            child.sequence = treenode.sequence
            if rng.random() < q:
                child.sequence = "T"
            child.terminated = False
            treenode.add_child(child)


def fitness_function(seq_list):
    r"""p and q parameters as a function of sequence"""
    return [(0.9, 0.5) if seq == "A" else (0.5, 0.5) for seq in seq_list]


class TestGC(unittest.TestCase):
    def setUp(self):
        self.T = 5
        self.N0 = 10
        self.sequence = "A"
        success = False
        while not success:
            try:
                self.gc = GC(
                    self.sequence, subtree_simulator, fitness_function, N0=self.N0
                )
                self.gc.simulate(self.T)
                success = True
            except RuntimeError:
                pass

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
