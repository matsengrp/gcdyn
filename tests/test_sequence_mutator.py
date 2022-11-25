import pytest

from gcdyn import bdms, mutators

# Note that some fixtures have been moved to conftest.py so they are more widely
# available.


def test_mutator():
    node = bdms.TreeNode(x="AGCT")
    mutator = mutators.SillySequenceMutator()
    mutator.mutate(node)


def test_mutator_in_tree():
    tree = bdms.TreeNode(x="AGCT")
    for seed in range(1000):
        try:
            tree.evolve(
                5,
                birth_rate=bdms.ConstantStringResponse(1),
                death_rate=bdms.ConstantStringResponse(1),
                mutation_rate=bdms.ConstantStringResponse(1),
                mutator=mutators.SillySequenceMutator(),
                min_survivors=20,
                seed=seed,
            )
            break
        except bdms.TreeError:
            continue
    assert False
