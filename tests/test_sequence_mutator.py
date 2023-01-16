import pytest
import pandas as pd

from gcdyn import bdms, gpmap, mutators, responses, utils

# Note that some fixtures have been moved to conftest.py so they are more widely
# available.


@pytest.fixture
def uniform_mutator():
    return mutators.UniformMutator()


@pytest.fixture
def gp_map():
    return gpmap.ConstantGPMap(1.0)


def test_simple_fivemer_contexts():
    sequence = "ACTGCA"
    correct_contexts = ["ACTGC", "CTGCA"]
    assert utils.simple_fivemer_contexts(sequence) == correct_contexts


def test_sequence_context_mutation_response(mk_rs5nf_mutability):
    node = bdms.TreeNode()
    node.sequence = "ACTGCA"
    seq_resp = responses.SequenceContextMutationResponse(
        mk_rs5nf_mutability, utils.simple_fivemer_contexts
    )
    correct_mutabilty = 0.000251042867724124 + 0.00233425857869025
    assert correct_mutabilty == pytest.approx(seq_resp(node))


def test_uniform_mutator(uniform_mutator, gp_map):
    node = bdms.TreeNode()
    node.sequence = "AGCT"
    node.x = gp_map(node.sequence)
    mutator = mutators.SequencePhenotypeMutator(uniform_mutator, gp_map)
    mutator.mutate(node)


def test_mutator_in_tree(uniform_mutator, gp_map):
    tree = bdms.TreeNode()
    tree.sequence = "AGCT"
    tree.x = gp_map(tree.sequence)
    for seed in range(1000):
        try:
            tree.evolve(
                5,
                mutator=mutators.SequencePhenotypeMutator(uniform_mutator, gp_map),
                min_survivors=20,
                seed=seed,
            )
            break
        except bdms.TreeError:
            continue
    print([node.sequence for node in tree.iter_leaves()])
