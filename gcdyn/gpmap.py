r"""
Genotype-phenotype map :py:class:`GPMap`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Abstract base class for defining generic genotype-phenotype maps.
Some concrete child classes are included.
"""
from abc import ABC, abstractmethod
from typing import Any


class GPMap(ABC):
    r"""Abstract base class genotype-phenotype maps."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __call__(self, sequence: str) -> float:
        r"""Evaluate a GPMap on a given DNA sequence.

        Args:
            sequence: A DNA sequence.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in vars(self).items() if not key.startswith('_'))})"


class ConstantGPMap(GPMap):
    r"""A map that evaluates to a constant.

    Args:
        constant: The constant returned by this GPMap.
    """

    def __init__(self, constant: float, *args: Any, **kwargs: Any) -> None:
        self.constant = constant

    def __call__(self, sequence: str) -> float:
        r"""Evaluate a GPMap on a given DNA sequence.

        Args:
            sequence: A DNA sequence.
        """
        return self.constant
