r"""Mapping of sequence to phenotype."""

import torch
import pandas


class Phenotype:
    r"""A class to map sequence to phenotype.

    Args:
        model_path: a path to a torchdms model
        phenotype_names: the names of the phenotypes
    """

    def __init__(self, model_path: str, phenotype_names: list[str]):
        self.model = torch.load(model_path)
        self.names = phenotype_names

    def evaluate(self, seqs: list[str]):
        "Evaluate phenotype given a list of sequences."
        aa_seq_one_hot = torch.stack([self.model.seq_to_binary(seq) for seq in seqs])
        labeled_evaluation = pandas.DataFrame(self.model(aa_seq_one_hot).detach().numpy(), columns=self.names)
        return labeled_evaluation
