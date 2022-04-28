from jax import random

from gcdyn.gc_tree import GC_tree


class GC_forest:
    def __init__(self, T, key, params, n_trees):
        key, _ = random.split(key)
        self.params = params

        self.trees = []
        self.create_trees(T, key, n_trees)

    def create_trees(self, T, key, n_trees):
        for i in range(n_trees):
            key, _ = random.split(key)
            tree = GC_tree(T, key, self.params)
            self.trees.append(tree)

    def draw_forest(self):
        for i, tree in enumerate(self.trees, 1):
            file_name = f"tree {i}"
            print(file_name)
            tree.draw_tree()
