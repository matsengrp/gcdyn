r"""Genotype-phenotype map :py:class:`GPMap`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Abstract base class for defining generic genotype-phenotype (GP) maps.
Some concrete child classes are included.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd
from Bio.Seq import Seq


class GPMap(ABC):
    r"""Abstract base class genotype-phenotype map."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __call__(self, sequence: str) -> float:
        r"""Evaluate GP map on a given DNA sequence.

        Args:
            sequence: A DNA sequence.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in vars(self).items() if not key.startswith('_'))})"


class ConstantGPMap(GPMap):
    r"""A GP map that evaluates to a constant.

    Args:
        constant: The constant returned by this GP map.
    """

    def __init__(self, constant: float) -> None:
        self.constant = constant

    def __call__(self, sequence: str) -> float:
        return self.constant


class AdditiveGPMap(GPMap):
    r"""A GP map with additive effects over the amino acids in a given DNA
    sequence.

    Args:
        mutation_effects: A :py:class:`pandas.DataFrame` of phenotypic effects of amino acid states at all sites, indexed by (0-based) sites in the translated sequence and columns corresponding to amino acid identity.
        nonsense_phenotype: The phenotype of a sequence with a nonsense mutation (stop codon).
    """

    def __init__(
        self, mutation_effects: pd.DataFrame, nonsense_phenotype: float = np.nan
    ) -> None:
        if not isinstance(mutation_effects.index, pd.RangeIndex):
            raise ValueError("mutation_effects.index must be a pandas.RangeIndex.")
        if mutation_effects.index.start != 0:
            raise ValueError("mutation_effects.index must start at 0.")
        if not set(mutation_effects.columns) == set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError(
                "mutation_effects.columns must be the 20 amino acids letters."
            )
        self.mutation_effects = mutation_effects.to_numpy()
        self.aa_indexer = {aa: i for i, aa in enumerate(mutation_effects.columns)}
        self.nonsense_phenotype = nonsense_phenotype

    def __call__(self, sequence: str) -> float:
        aa_sequence = Seq(sequence).translate()
        if "*" in aa_sequence:
            return self.nonsense_phenotype
        if len(aa_sequence) != len(self.mutation_effects):
            raise ValueError(
                "amino acid sequence length does not match length of mutation_effects DataFrame."
            )
        return sum(
            self.mutation_effects[i, self.aa_indexer[aa]]
            for i, aa in enumerate(aa_sequence)
        )
