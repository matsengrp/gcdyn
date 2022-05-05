from jax import random

from gcdyn.gc_tree import GC_tree
from gcdyn.parameters import Parameters


class GC_forest:
    r"""A class that represents a collection of GC tree

    Args:
        T (float): simulation sampling time
        key (int): seed to generate random key
        params (Parameters): model parameters
        n_trees (int): number of GC trees in GC forest
    """

    def __init__(self, T: float, key: int, params: Parameters, n_trees: int):
        key, _ = random.split(key)
        self.params = params

        self.forest: list[GC_tree] = []
        self.create_trees(T, key, n_trees)

    def create_trees(self, T: float, key: int, n_trees: int):
        r"""Create n_Trees number of GC tree

        Args:
            T (float): simulation sampling time
            key (int): seed to generate random key
            n_trees (int): number of GC trees to create
        """
        for i in range(n_trees):
            key, _ = random.split(key)
            tree = GC_tree(T, key, self.params)
            self.forest.append(tree)

    def draw_forest(self, folder_name: str = None):
        r"""Visualizes the forest

        If folder_name is given, tree visualizations are saved in the folder.
        If not, tree visualizations are randered to the notebook.

        Args:
            folder_name (str): name of the output folder of the tree visualizations. Defaults to None.
        """
        for i, tree in enumerate(self.forest, 1):
            if folder_name is None:
                tree.draw_tree()
            else:
                tree.draw_tree(folder_name + f"tree {i}")
