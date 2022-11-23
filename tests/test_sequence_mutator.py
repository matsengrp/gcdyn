import pytest
import pandas as pd

from gcdyn import bdms, gpmap, mutators, responses, utils

# Note that some fixtures have been moved to conftest.py so they are more widely
# available.

# TODO what tests do we really want?


@pytest.fixture
def uniform_mutator():
    return mutators.UniformMutator()


@pytest.fixture
def mutability(MK_RS5NF_mutability_path):
    return pd.read_csv(MK_RS5NF_mutability_path, index_col=0).squeeze("columns")


@pytest.fixture
def substitution(MK_RS5NF_substitution_path):
    return pd.read_csv(MK_RS5NF_substitution_path, index_col=0)


@pytest.fixture
def replay_seq_to_contexts():
    chain_2_start_idx = 336
    return lambda seq: utils.padded_fivemer_contexts_of_paired_sequences(
        seq, chain_2_start_idx
    )


@pytest.fixture
def replay_mutator(mutability, substitution, replay_seq_to_contexts):
    return mutators.ContextMutator(
        mutability=mutability,
        substitution=substitution,
        seq_to_contexts=replay_seq_to_contexts,
    )


@pytest.fixture
def replay_mutation_response(mutability, substitution, replay_seq_to_contexts):
    return responses.SequenceContextMutationResponse(mutability, replay_seq_to_contexts)


@pytest.fixture
def gp_map():
    return gpmap.ConstantGPMap(1.0)


def test_simple_fivemer_contexts():
    sequence = "ACTGCA"
    correct_contexts = ["ACTGC", "CTGCA"]
    assert utils.simple_fivemer_contexts(sequence) == correct_contexts


def test_sequence_context_mutation_response(mutability):
    node = bdms.TreeNode()
    node.sequence = "ACTGCA"
    seq_resp = responses.SequenceContextMutationResponse(
        mutability, utils.simple_fivemer_contexts
    )
    correct_mutabilty = 0.000251042867724124 + 0.00233425857869025
    assert correct_mutabilty == pytest.approx(seq_resp(node))


def test_uniform_mutator(uniform_mutator, gp_map):
    node = bdms.TreeNode()
    node.sequence = "AGCT"
    node.x = gp_map(node.sequence)
    mutator = mutators.SequencePhenotypeMutator(uniform_mutator, gp_map)
    mutator.mutate(node)


def test_fivemer_mutator(replay_mutator, replay_naive):
    node = bdms.TreeNode()
    node.sequence = replay_naive
    for i in range(10):
        replay_mutator.mutate(node)


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


def test_fivemer_mutator_in_tree(
    replay_mutator, gp_map, replay_naive, replay_mutation_response
):
    tree = bdms.TreeNode()
    tree.sequence = replay_naive
    tree.x = gp_map(tree.sequence)
    for seed in range(1000):
        try:
            tree.evolve(
                5,
                mutator=mutators.SequencePhenotypeMutator(replay_mutator, gp_map),
                mutation_rate=replay_mutation_response,
                min_survivors=20,
                seed=seed,
            )
            break
        except bdms.TreeError:
            continue
