"""
Various things relevant for the replay experiment.
"""
import os
import sys

import pandas as pd

from gcdyn import mutators, responses, utils

# TODO Will can you double check this naive sequence?

naive_sequence = "GAGGTGCAGCTTCAGGAGTCAGGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTGTCACTGGCGACTCCATCACCAGTGGTTACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGGTACATAAGCTACAGTGGTAGCACTTACTACAATCCATCTCTCAAAAGTCGAATCTCCATCACTCGAGACACATCCAAGAACCAGTACTACCTGCAGTTGAATTCTGTGACTACTGAGGACACAGCCACATATTACTGTGCAAGGGACTTCGATGTCTGGGGCGCAGGGACCACGGTCACCGTCTCCTCAGACATTGTGATGACTCAGTCTCAAAAATTCATGTCCACATCAGTAGGAGACAGGGTCAGCGTCACCTGCAAGGCCAGTCAGAATGTGGGTACTAATGTAGCCTGGTATCAACAGAAACCAGGGCAATCTCCTAAAGCACTGATTTACTCGGCATCCTACAGGTACAGTGGAGTCCCTGATCGCTTCACAGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAATGTGCAGTCTGAAGACTTGGCAGAGTATTTCTGTCAGCAATATAACAGCTATCCTCTCACGTTCGGCTCGGGGACTAAGCTAGAAATAAAA"

chain_2_start_idx = 336

gcdyn_data_dir = os.path.join(os.path.dirname(sys.modules["gcdyn"].__file__), "data")


mutability = pd.read_csv(
    os.path.join(gcdyn_data_dir, "MK_RS5NF_mutability.csv"), index_col=0
).squeeze("columns")

substitution = pd.read_csv(
    os.path.join(gcdyn_data_dir, "MK_RS5NF_substitution.csv"), index_col=0
)


def seq_to_contexts(seq):
    return utils.padded_fivemer_contexts_of_paired_sequences(seq, chain_2_start_idx)


mutator = mutators.ContextMutator(
    mutability=mutability,
    substitution=substitution,
    seq_to_contexts=seq_to_contexts,
)

mutation_response = responses.SequenceContextMutationResponse(
    mutability, seq_to_contexts
)
