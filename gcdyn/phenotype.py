r"""Uses PyTorch for Deep Mutational Scanning (``torchdms``) model to predict affinity of Ig heavy and light chain DNA sequences to presented antigen."""
from __future__ import annotations
import pandas as pd
import torch
from Bio.Seq import Seq


# phenotype evaluation:


class DMSPhenotype:
    """Defines a set of parameters used to go from DNA sequence of heavy and
    light chain to KD values.

    Args:
        igh_frame: frame for translation of Ig heavy chain
        igk_frame: frame for translation of Ig light chain
        igk_idx: index of Ig light chain starting position
        naive_sites_path: path to CSV lookup table for converting from scFv CDS indexed site numbering to heavy/light chain IMGT numbering
        model_path: path to ``torchdms`` model for heavy and light chain
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
        self.pos_df = pd.read_csv(
            naive_sites_path,
            dtype=dict(site=pd.Int16Dtype()),
            index_col="site_scFv",
        )
        self.model = torch.load(model_path)
        self.tdms_phenotypes = tdms_phenotypes
        self.log10_naive_KD = log10_naive_KD

    def _seq_aa_tdms(self, nt_seqs: list[str]) -> list[str]:
        """Produces ``torchdms``-inferred Igh and Igk amino acid sequence from
        a list of DNA sequences.

        Args:
            nt_seqs: list of DNA sequences of length of 657 nt

        Returns:
            aa_seqs: amino acid sequences translated based on known features
        """
        # make a prediction for each of the observed sequences
        aa_seqs = []
        for nt_seq in nt_seqs:
            if len(nt_seq) != 657:
                raise Exception(f"all sequences must be 657 nt, len = {len(nt_seq)}")
            igh_aa = self._aa(nt_seq[: self.igk_idx], self.igh_frame)
            igk_aa = self._aa(nt_seq[self.igk_idx :], self.igk_frame)
            aa_tdms = self.pos_df.amino_acid.copy()
            aa_tdms.iloc[self.pos_df.chain == "H"] = igh_aa
            # note: replay light chains are shorter than dms seq by one aa
            aa_tdms.iloc[
                (self.pos_df.chain == "L") & (self.pos_df.index < self.pos_df.index[-1])
            ] = igk_aa
            aa_seqs.append("".join(aa_tdms))
        return aa_seqs

    def calculate_KD(self, nt_seqs: list[str]) -> list[float]:
        r"""Produces KD values for a collection of sequences using ``torchdms`` model that can produce ``delta_log10_KD`` values.

        Args:
            nt_seqs: list of DNA sequences

        Returns:
            kd_values: predicted KD based on ``torchdms`` model for each sequence
        """

        seqs = self._seq_aa_tdms(nt_seqs)
        aa_seq_one_hot = torch.stack([self.model.seq_to_binary(seq) for seq in seqs])
        try:
            labeled_evaluation = pd.DataFrame(
                self.model(aa_seq_one_hot).detach().numpy(),
                columns=self.tdms_phenotypes,
            )
        except ValueError:
            print("Incorrect number of column labels for phenotype data")

        kd_values = list(
            10 ** (delta_log10_KD + self.log10_naive_KD)
            for delta_log10_KD in labeled_evaluation["delta_log10_KD"]
        )
        return kd_values

    @staticmethod
    def _aa(sequence: str, frame: int) -> Seq:
        """Amino acid translation of nucleotide sequence in frame 1, 2, or 3.

        Args:
            sequence: DNA sequence
            frame: frame for translation

        Returns:
            aa_seq: translated sequence
        """
        return Seq(
            sequence[
                (frame - 1) : (frame - 1 + (3 * ((len(sequence) - (frame - 1)) // 3)))
            ]
        ).translate()
