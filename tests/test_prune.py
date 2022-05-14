import jax.numpy as np
from jax import random

from gcdyn.gc_tree import GC_tree
from gcdyn.gc_forest import GC_forest
from gcdyn.parameters import Parameters


def test_prune_tree():
    T = 3
    seed = 0
    key = random.PRNGKey(seed)

    # response function parameters
    θ = np.array([3, 1, 0], dtype=float)
    # death rate
    μ = 1
    # mutation rate
    m = 1
    # sampling efficiency
    ρ = 0.5

    params = Parameters(θ, μ, m, ρ)

    key, _ = random.split(key)
    tree = GC_tree(T, key, params)
    original_sampled = set([node for node in tree.tree.traverse() if node.event == "sampled"])
    tree.prune_tree()

    assert(all(node.event == "sampled" for node in tree.tree.traverse() if node.is_leaf()))
    assert(all(len(node.children) == 2 for node in tree.tree.traverse() if node.event == "birth"))
    assert(set([node for node in tree.tree.traverse() if node.event == "sampled"]) == original_sampled)


def test_prune_forest():
    n_trees = 5
    T = 3
    seed = 0
    key = random.PRNGKey(seed)

    # response function parameters
    θ = np.array([3, 1, 0], dtype=float)
    # death rate
    μ = 1
    # mutation rate
    m = 1
    # sampling efficiency
    ρ = 0.5

    params = Parameters(θ, μ, m, ρ)

    key, _ = random.split(key)
    forest = GC_forest(T, key, params, n_trees)
    forest.prune_forest()

    for tree in forest.forest:
        assert(all(node.event == "sampled" for node in tree.tree.traverse() if node.is_leaf()))
        assert(all(len(node.children) == 2 for node in tree.tree.traverse() if node.event == "birth"))
