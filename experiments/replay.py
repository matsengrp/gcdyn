"""
Various things relevant for the GC replay experiment.
"""
import os
import sys
import pandas as pd
from typing import Dict, List, Tuple

from gcdyn import utils

NAIVE_SEQUENCE = "GAGGTGCAGCTTCAGGAGTCAGGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTGTCACTGGCGACTCCATCACCAGTGGTTACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGGTACATAAGCTACAGTGGTAGCACTTACTACAATCCATCTCTCAAAAGTCGAATCTCCATCACTCGAGACACATCCAAGAACCAGTACTACCTGCAGTTGAATTCTGTGACTACTGAGGACACAGCCACATATTACTGTGCAAGGGACTTCGATGTCTGGGGCGCAGGGACCACGGTCACCGTCTCCTCAGACATTGTGATGACTCAGTCTCAAAAATTCATGTCCACATCAGTAGGAGACAGGGTCAGCGTCACCTGCAAGGCCAGTCAGAATGTGGGTACTAATGTAGCCTGGTATCAACAGAAACCAGGGCAATCTCCTAAAGCACTGATTTACTCGGCATCCTACAGGTACAGTGGAGTCCCTGATCGCTTCACAGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAATGTGCAGTCTGAAGACTTGGCAGAGTATTTCTGTCAGCAATATAACAGCTATCCTCTCACGTTCGGCTCGGGGACTAAGCTAGAAATAAAA"
"""The naive sequence used in the GC replay experiment."""

CHAIN_2_START_IDX = 336
"""The index of the first nucleotide of the light chain in the naive sequence."""

gcdyn_data_dir = os.path.join(os.path.dirname(sys.modules["gcdyn"].__file__), "data")


def mutability(file: str = "MK_RS5NF_mutability.csv") -> pd.Series:
    """The mutability of each position in the naive sequence.

    Args:
        file: The file to read the mutability from.
    """
    return pd.read_csv(os.path.join(gcdyn_data_dir, file), index_col=0).squeeze(
        "columns"
    )


def substitution(file: str = "MK_RS5NF_substitution.csv") -> pd.DataFrame:
    """The substitution matrix for the naive sequence.

    Args:
        file: The file to read the substitution matrix from.
    """
    return pd.read_csv(os.path.join(gcdyn_data_dir, file), index_col=0)


def seq_to_contexts(seq) -> Tuple[str, ...]:
    """Convert a replay BCR sequence to a tuple of 5-mer contexts.

    Args:
        seq: The sequence to convert.
    """
    return utils.padded_fivemer_contexts_of_paired_sequences(seq, CHAIN_2_START_IDX)


def dms(
    file: str = "https://media.githubusercontent.com/media/jbloomlab/Ab-CGGnaive_DMS/main/results/final_variant_scores/final_variant_scores.csv",
) -> Dict[str, pd.DataFrame]:
    """The DMS data for the GC replay experiment.

    Args:
        file: The file to read the DMS data from.

    Returns:
        A dictionary with the DMS data for each of the three phenotypes.
    """
    dms_df = pd.read_csv(file, index_col="mutation")
    # remove linker sites
    dms_df = dms_df[dms_df.chain != "link"]

    bind_df = (
        dms_df.pivot(index="position", columns="mutant", values="delta_bind_CGG")
        .reset_index()
        .drop(columns="position")
    )
    expr_df = (
        dms_df.pivot(index="position", columns="mutant", values="delta_expr")
        .reset_index()
        .drop(columns="position")
    )
    psr_df = (
        dms_df.pivot(index="position", columns="mutant", values="delta_psr")
        .reset_index()
        .drop(columns="position")
    )

    # the DMS has one additional codon wrt replay sequence
    bind_df.drop(index=bind_df.index[-1], inplace=True)
    expr_df.drop(index=expr_df.index[-1], inplace=True)
    psr_df.drop(index=psr_df.index[-1], inplace=True)

    # replace NaNs with 0 (some single mutants are missing in DMS data)
    bind_df.fillna(0, inplace=True)
    expr_df.fillna(0, inplace=True)
    psr_df.fillna(0, inplace=True)

    return dict(affinity=bind_df, expr=expr_df, psr=psr_df)
