"""
Various things relevant for the replay experiment.
"""
import os
import sys
import pandas as pd

from gcdyn import poisson, utils

NAIVE_SEQUENCE = "GAGGTGCAGCTTCAGGAGTCAGGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTGTCACTGGCGACTCCATCACCAGTGGTTACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGGTACATAAGCTACAGTGGTAGCACTTACTACAATCCATCTCTCAAAAGTCGAATCTCCATCACTCGAGACACATCCAAGAACCAGTACTACCTGCAGTTGAATTCTGTGACTACTGAGGACACAGCCACATATTACTGTGCAAGGGACTTCGATGTCTGGGGCGCAGGGACCACGGTCACCGTCTCCTCAGACATTGTGATGACTCAGTCTCAAAAATTCATGTCCACATCAGTAGGAGACAGGGTCAGCGTCACCTGCAAGGCCAGTCAGAATGTGGGTACTAATGTAGCCTGGTATCAACAGAAACCAGGGCAATCTCCTAAAGCACTGATTTACTCGGCATCCTACAGGTACAGTGGAGTCCCTGATCGCTTCACAGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAATGTGCAGTCTGAAGACTTGGCAGAGTATTTCTGTCAGCAATATAACAGCTATCCTCTCACGTTCGGCTCGGGGACTAAGCTAGAAATAAAA"

CHAIN_2_START_IDX = 336

gcdyn_data_dir = os.path.join(os.path.dirname(sys.modules["gcdyn"].__file__), "data")


mutability = pd.read_csv(
    os.path.join(gcdyn_data_dir, "MK_RS5NF_mutability.csv"), index_col=0
).squeeze("columns")

substitution = pd.read_csv(
    os.path.join(gcdyn_data_dir, "MK_RS5NF_substitution.csv"), index_col=0
)


def seq_to_contexts(seq):
    return utils.padded_fivemer_contexts_of_paired_sequences(seq, CHAIN_2_START_IDX)


mutation_response = poisson.SequenceContextMutationResponse(
    mutability, seq_to_contexts
)

dms_df = pd.read_csv("https://media.githubusercontent.com/media/jbloomlab/Ab-CGGnaive_DMS/main/results/final_variant_scores/final_variant_scores.csv", index_col="mutation")
# remove linker sites
dms_df = dms_df[dms_df.chain != "link"]

bind_df = dms_df.pivot(index="position", columns="mutant", values="delta_bind_CGG").reset_index().drop(columns="position")
expr_df = dms_df.pivot(index="position", columns="mutant", values="delta_expr").reset_index().drop(columns="position")
psr_df = dms_df.pivot(index="position", columns="mutant", values="delta_psr").reset_index().drop(columns="position")

# the DMS has one additional codon wrt replay sequence
bind_df.drop(index=bind_df.index[-1], inplace=True)
expr_df.drop(index=expr_df.index[-1], inplace=True)
psr_df.drop(index=psr_df.index[-1], inplace=True)

# replace NaNs with 0 (some single mutants are missing in DMS data)
bind_df.fillna(0, inplace=True)
expr_df.fillna(0, inplace=True)
psr_df.fillna(0, inplace=True)
