r"""Mapping of sequence to phenotype."""

import torch


class Phenotype:
    r"""A class to map sequence to phenotype.

    Args:
        model_path: a path to a torchdms model
        TODO phenotype_names: the names of the phenotypes
    """

    def __init__(self, model_path: str):
        self.model = torch.load(model_path)

    def evaluate(self, seqs: list[str]):
        "Evaluate phenotype given a list of sequences."
        aa_seq_one_hot = torch.stack([self.model.seq_to_binary(seq) for seq in seqs])
        # TODO: use the names as column labels in pandas, checking to make sure that the
        # number of columns output by the model is the same as the length of
        # phenotype_names .
        return self.model(aa_seq_one_hot).detach().numpy()
