from jax import random

import gc_tree


class GC_forest:

    def __init__(self, T, key, θ, μ, m, ρ, n_trees):
        key, _ = random.split(key)
        self.θ = θ
        self.μ = μ
        self.m = m
        self.ρ = ρ

        self.trees = []
        self.log_likelihood = 0
        self.create_trees(T, key, n_trees)
    
    def create_trees(self, T, key, n_trees):
        for i in range(n_trees):
            key, _ = random.split(key)
            tree = gc_tree.GC_tree(T, key, self.θ, self.μ, self.m, self.ρ)
            self.trees.append(tree)
            self.log_likelihood += tree.log_likelihood()
    
    def draw_forest(self):
        for i, tree in enumerate(self.trees, 1):
            file_name = f"tree {i}"
            print(file_name)
            tree.draw_tree(file_name)