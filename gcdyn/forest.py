r"""Forests of GC trees"""

from jax import random

from gcdyn.tree import Tree
from gcdyn.parameters import Parameters


class Forest:
    r"""A class that represents a collection of GC tree

    Args:
        T: simulation sampling time
        seed: random seed
        params: model parameters
        n_trees: number of GC trees in GC forest
    """

    def __init__(self, T: float, seed: int, params: Parameters, n_trees: int):
        self.params = params

        self.forest: list[Tree] = []
        self.create_trees(T, random.PRNGKey(seed), n_trees)

    def create_trees(self, T: float, key: random.PRNGKeyArray, n_trees: int):
        r"""Create n_Trees number of GC tree

        Args:
            T: simulation sampling time
            key: random key
            n_trees: number of GC trees to create
        """
        for i in range(n_trees):
            key, _ = random.split(key)
            tree = Tree(T, key, self.params)
            self.forest.append(tree)

    def draw_forest(self, folder_name: str = None):
        r"""Visualizes the forest

        If folder_name is given, tree visualizations are saved in the folder.
        If not, tree visualizations are rendered to the notebook.

        Args:
            folder_name: name of the output folder of the tree visualizations. Defaults to None.
        """
        for i, tree in enumerate(self.forest, 1):
            if folder_name is None:
                tree.draw_tree()
            else:
                tree.draw_tree(folder_name + f"tree {i}")

    def prune(self):
        for tree in self.forest:
            tree.prune()
