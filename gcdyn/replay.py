from __future__ import annotations
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq


# From gcreplay-tools (https://github.com/matsengrp/gcreplay/blob/main/nextflow/bin/gcreplay-tools.py):


def fasta_to_seq_list(f: str) -> list(str):
    """simply convert a fasta to a list of sequences.

    Args:
        f: path to fasta file
    Returns:
        seqs: list of sequences from fasta, excluding any with id 'naive'
    """
    seqs = []
    try:
        with open(f) as fasta_file:
            for seq_record in SeqIO.parse(fasta_file, "fasta"):  # (generator)
                if str(seq_record.id) != "naive":
                    seqs.append(str(seq_record.seq))
    except OSError:
        print("Unable to open {0}".format(f))
    return seqs


def aa(sequence: str, frame: int) -> Seq:
    """Amino acid translation of nucleotide sequence in frame 1, 2, or 3.

    Args:
        sequence: DNA sequence
        frame: frame for translation

    Returns:
        aa_seq: translated sequence
    """
    return Seq(
        sequence[(frame - 1) : (frame - 1 + (3 * ((len(sequence) - (frame - 1)) // 3)))]
    ).translate()


def read_sites_file(naive_sites_path: str) -> pd.DataFrame:
    """Read the sites file from csv.

    Args:
        naive_sites_path: path to naive sites CSV with ``site_scFv`` column

    Returns:
        DataFrame of positions for heavy and light chains
    """
    # collect the correct sites df
    pos_df = pd.read_csv(
        naive_sites_path,
        dtype=dict(site=pd.Int16Dtype()),
        index_col="site_scFv",
    )
    return pos_df


# phenotype evaluation:


class ReplayPhenotype:
    """Defines a set of parameters used to go from DNA sequence of antibody
    sequences to KD values.

    Args:
        igh_frame: frame for translation of Ig heavy chain
        igk_frame: frame for translation of Ig light chain
        igk_idx: index of Ig light chain starting position
        naive_sites_path: path to CSV lookup table for converting from scFv CDS indexed site numbering to heavy/light chain IMGT numbering
        model_path: path to ``torchdms`` model for antibody sequences
        tdms_phenotypes: names of phenotype values produced by passed-in ``torchdms`` model (``delta_log10_KD`` expected as a phenotype)
        log10_naive_KD: KD of naive Ig
    """

    def __init__(
        self,
        igh_frame: int,
        igk_frame: int,
        igk_idx: int,
        naive_sites_path: str,
        model_path: str,
        tdms_phenotypes: list[str],
        log10_naive_KD: float,
    ):
        self.igh_frame = igh_frame
        self.igk_frame = igk_frame
        self.igk_idx = igk_idx
        self.pos_df = read_sites_file(naive_sites_path)
        self.model = torch.load(model_path)
        self.tdms_phenotypes = tdms_phenotypes
        self.log10_naive_KD = log10_naive_KD

    def seq_df_tdms(self, seq_list: list[str]) -> pd.DataFrame:
        """Produces DataFrame with amino acid sequence from a list of DNA
        sequences.

        Args:
            seq_list: list of DNA sequences of length of 657 nt

        Returns:
            seqs_df: DataFrame with columns for DNA sequence (``seq``), and amino acid sequence (``aa_sequence``)
        """
        if seq_list is not None:
            seqs_df = pd.DataFrame({"seq": seq_list, "aa_sequence": ""})
        else:
            raise Exception("list of sequences must be given")

        # make a prediction for each of the observed sequences
        for idx, row in seqs_df.iterrows():
            if len(row.seq) != 657:
                raise Exception(
                    "all sequences must be 657 nt, len = {0}".format(len(row.seq))
                )
            # translate heavy and light chains
            igh_aa = aa(row.seq[: self.igk_idx], self.igh_frame)
            igk_aa = aa(row.seq[self.igk_idx :], self.igk_frame)

            # Make the aa seq for tdms
            aa_tdms = self.pos_df.amino_acid.copy()
            aa_tdms.iloc[self.pos_df.chain == "H"] = igh_aa
            # note: replay light chains are shorter than dms seq by one aa
            aa_tdms.iloc[
                (self.pos_df.chain == "L") & (self.pos_df.index < self.pos_df.index[-1])
            ] = igk_aa
            aa_tdms_seq = "".join(aa_tdms)
            seqs_df.loc[idx, "aa_sequence"] = aa_tdms_seq

        return seqs_df

    def calculate_KD(self, seq_list: list[str]) -> pd.DataFrame:
        r"""Produces KD values for a collection of sequences using ``torchdms`` model.

        Args:
            seq_list: list of DNA sequences

        Returns:
            seqs_df: a ``DataFrame`` with columns for ``aa_sequence``, each of ``tdms_phenotypes``, and ``KD``
        """

        seqs_df = self.seq_df_tdms(seq_list)
        seqs = seqs_df["aa_sequence"]
        aa_seq_one_hot = torch.stack([self.model.seq_to_binary(seq) for seq in seqs])
        try:
            labeled_evaluation = pd.DataFrame(
                self.model(aa_seq_one_hot).detach().numpy(),
                columns=self.tdms_phenotypes,
            )
        except ValueError:
            print("Incorrect number of column labels for phenotype data")
        labeled_evaluation["KD"] = 10 ** (
            labeled_evaluation["delta_log10_KD"] + self.log10_naive_KD
        )
        return labeled_evaluation["KD"]
