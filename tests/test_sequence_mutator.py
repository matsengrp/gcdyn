import pytest
import numpy as np

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
    correct_contexts = ("ACTGC", "CTGCA")
    assert utils.simple_fivemer_contexts(sequence) == correct_contexts


def test_paired_fivemer_contexts():
    pairseqs = ["ACTGCA", "GCATCA"]
    def ctxfn(s): return tuple([s[(i - 2) : (i + 3)] for i in range(2, len(s) - 2)])
    correct_contexts = tuple([c for s in pairseqs for c in ctxfn("NN%sNN" % s)])
    node = bdms.TreeNode()
    node.sequence = ''.join(pairseqs)
    node.chain_2_start_idx = len(pairseqs[0])
    assert utils.node_contexts(node) == correct_contexts


def test_sequence_context_mutation_response(mk_rs5nf_mutability):
    node = bdms.TreeNode()
    node.sequence = "ACTGCA"
    seq_resp = poisson.SequenceContextMutationResponse(mk_rs5nf_mutability)
    correct_mutabilty = 0.000251042867724124 + 0.00233425857869025
    assert seq_resp(node) == pytest.approx(correct_mutabilty)


def test_uniform_mutator(uniform_mutator, gp_map):
    node = bdms.TreeNode()
    node.sequence = "AGCT"
    node.x = gp_map(node.sequence)
    mutator = mutators.SequencePhenotypeMutator(uniform_mutator, gp_map)
    mutator.mutate(node)

def test_replay_context_mutation(mk_rs5nf_mutability, replay_subst, gp_map, naive_seq, naive_contexts, chain_2_start_idx):
    node = bdms.TreeNode()
    node.sequence = naive_seq
    node.chain_2_start_idx = chain_2_start_idx
    node.x = gp_map(naive_seq)
    assert utils.node_contexts(node) == naive_contexts

    seq_resp = poisson.SequenceContextMutationResponse(mk_rs5nf_mutability)
    correct_mutabilty = 0.5854612215719815
    assert seq_resp(node) == pytest.approx(correct_mutabilty)

    rng = np.random.default_rng(seed=0)
    n_muts = 30
    mutator = mutators.SequencePhenotypeMutator(
        mutators.ContextMutator(mutability=mk_rs5nf_mutability, substitution=replay_subst),
        gp_map,
    )
    for _ in range(n_muts):
        mutator.mutate(node, seed=rng)
    correct_mutated_seq = 'GAGGTGCAGGTTCAGGAGTCAAGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTATCACTGGCGACTCCATCACCAGTGGTTACTGGAACTAGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGATACATACGCTACAGTGGTAACACTTACTACAATCCATCTCTCAAAAGTCGAATCTACATCACTCGAGACACATCCAAGAACCAGTATTACCTGCAGTTGAATTCTGTGACTACTGAGGACACAGCCACAGATTACTGTGCTAGGGACTTCGATGTCTGGGGCGCAGGGACCACGGTCACTGTCTCCTCAGACATTGTGATGACTCAGTCTCAAAATTTCATGTCCACATCGGTAGGAGACAGGGTCAACGTCACCTGCAAGGCCAGTCAGAATGTGGATATTAATGTAGCCTGGAATCAACAGAAACCAGGGCGATCTCCTAAGCCACTGATTTAGTCGGCATCCTACAGGTACAGTGGAGTCCCTGATCGCTTCACAGGCAGTGGATCTGGGGCAGATTTCACTCTCACCATCACCACTGTGCAGTCTGAAGACTTGGCAGAGAATTTCTGTTAGCACTATAACAGCTATCCTCTCACGTTCGACTCGGGGACTAAGCTAGAAATATAA'
    print('         naive: %s' % naive_seq)
    print('          mutd: %s' % utils.color_mutants(naive_seq, node.sequence))
    print('  correct mutd: %s' % utils.color_mutants(naive_seq, correct_mutated_seq))
    assert node.sequence == correct_mutated_seq
