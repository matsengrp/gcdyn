from __future__ import annotations
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq


# From gcreplay-tools (https://github.com/matsengrp/gcreplay/blob/main/nextflow/bin/gcreplay-tools.py):


def fasta_to_df(f):
    """simply convert a fasta to dataframe."""
    ids, seqs = [], []
    try:
        with open(f) as fasta_file:
            for seq_record in SeqIO.parse(fasta_file, "fasta"):  # (generator)
                if str(seq_record.id) != "naive":
                    ids.append(seq_record.id)
                    seqs.append(str(seq_record.seq))
    except OSError:
        print("Unable to open {0}".format(f))
    return pd.DataFrame({"id": ids, "seq": seqs})


def fasta_to_seq_list(f):
    """simply convert a fasta to dataframe."""
    seqs = []
    try:
        with open(f) as fasta_file:
            for seq_record in SeqIO.parse(fasta_file, "fasta"):  # (generator)
                if str(seq_record.id) != "naive":
                    seqs.append(str(seq_record.seq))
    except OSError:
        print("Unable to open {0}".format(f))
    return seqs


def aa(sequence, frame):
    """Amino acid translation of nucleotide sequence in frame 1, 2, or 3."""
    return Seq(
        sequence[(frame - 1) : (frame - 1 + (3 * ((len(sequence) - (frame - 1)) // 3)))]
    ).translate()


def read_sites_file(naive_sites_path: str):
    """Read the sites file from csv."""
    # collect the correct sites df from tylers repo
    pos_df = pd.read_csv(
        naive_sites_path,
        dtype=dict(site=pd.Int16Dtype()),
        index_col="site_scFv",
    )
    return pos_df


# phenotype evaluation:


class ReplayPhenotype:
    """Defines a set of parameters used to go from sequence to KD values."""

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
        self.naive_sites_path = naive_sites_path
        self.model_path = model_path
        self.tdms_phenotypes = tdms_phenotypes
        self.log10_naive_KD = log10_naive_KD

    def seq_df_tdms(self, seq_list: list[str] = None):
        if seq_list is not None:
            seqs_df = pd.DataFrame({"seq": seq_list})
        else:
            raise Exception("list of sequences must be given")

        # make a prediction for each of the observed sequences
        for idx, row in seqs_df.iterrows():

            # translate heavy and light chains
            igh_aa = aa(row.seq[: self.igk_idx], self.igh_frame)
            igk_aa = aa(row.seq[self.igk_idx :], self.igk_frame)

            # Make the aa seq for tdms
            pos_df = read_sites_file(self.naive_sites_path)
            aa_tdms = pos_df.amino_acid.copy()
            aa_tdms.iloc[pos_df.chain == "H"] = igh_aa
            # note: replay light chains are shorter than dms seq by one aa
            aa_tdms.iloc[
                (pos_df.chain == "L") & (pos_df.index < pos_df.index[-1])
            ] = igk_aa
            aa_tdms_seq = "".join(aa_tdms)
            seqs_df.loc[idx, "aa_sequence"] = aa_tdms_seq

        return seqs_df

    def calculate_KD_df(self, seq_list: list[str] = None):
        r"""Builds a `DataFrame` with KD values for a collection of sequences

        Args:
            seq_list: list of DNA sequences
        Returns:
            seqs_df: a `DataFrame` with columns for `aa_sequence`, each of `tdms_phenotypes`, `KD`
        """

        seqs_df = self.seq_df_tdms(seq_list)
        seqs = seqs_df["aa_sequence"]
        torchdms_model = torch.load(self.model_path)
        aa_seq_one_hot = torch.stack(
            [torchdms_model.seq_to_binary(seq) for seq in seqs]
        )
        try:
            labeled_evaluation = pd.DataFrame(
                torchdms_model(aa_seq_one_hot).detach().numpy(),
                columns=self.tdms_phenotypes,
            )
        except ValueError:
            print("Incorrect number of column labels for phenotype data")
        labeled_evaluation["KD"] = 10 ** (
            labeled_evaluation["delta_log10_KD"] + self.log10_naive_KD
        )
        seqs_df = seqs_df.merge(labeled_evaluation, left_index=True, right_index=True)
        return seqs_df

    def return_KD(self, seq_list: list[str] = None):
        """Returns a list of KD values for a collection of sequences."""
        seqs_df = self.calculate_KD_df(seq_list)
        return seqs_df["KD"]
