import pytest

from gcdyn import bdms, gpmap, mutators, poisson, utils

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
    seq_resp = poisson.SequenceContextMutationResponse(
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
