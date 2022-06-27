r"""Uses phenotype to determine fitness"""
from __future__ import annotations
from typing import Callable
from math import exp
import pandas as pd


class Fitness:
    r"""Class to determine fitness from phenotype for a collection of sequences

    Args:
        mapping_type: type of mapping function (defaults to linear)
        linfit_slope: slope for linear mapping between antigen bound and fitness
        maximum_Tfh: maximum Tfh help for sigmoidal mapping to fitness
        curve_steepness: logistic growth rate for sigmoidal mapping to fitness
        midpoint_antigen_bound: midpoint of antigen bound for sigmoidal mapping to fitness
        concentration_antigen: molar concentration of antigen to determine antigen bound
    """

    def __init__(
        self,
        mapping_type: str = "linear",
        linfit_slope: float = 1,
        maximum_Tfh: float = 4,
        curve_steepness: float = 10,
        midpoint_antigen_bound: float = 0.5,
        concentration_antigen: float = 10 ** (-9),
    ):
        if mapping_type.lower() == "linear":
            self.fitness_func = self.linear_fitness
        elif mapping_type.lower() == "sigmoid":
            self.fitness_func = self.sigmoidal_fitness
        else:
            raise Exception("Only linear and sigmoid are acceptable mapping types")
        self.linfit_slope = linfit_slope
        self.maximum_Tfh = maximum_Tfh
        self.curve_steepness = curve_steepness
        self.midpoint_antigen_bound = midpoint_antigen_bound
        self.concentration_antigen = concentration_antigen

    def frac_antigen_bound(self, KD_values: list[float]):
        r"""Determine the fraction of antigen bound from a list of values using the Hill equation."""
        thetas = []
        for KD in KD_values:
            # Hill equation with n = 1:
            theta = self.concentration_antigen / (KD + self.concentration_antigen)
            thetas.append(theta)
        return thetas

    def linear_fitness(self, seq_df: pd.DataFrame):
        r"""Combines methods to get the antigen bound, Tfh help, and fitness
        from the KD using a linear model."""
        seq_df["frac_antigen_bound"] = self.frac_antigen_bound(seq_df["KD"])
        seq_df["fitness"] = seq_df["frac_antigen_bound"] * self.linfit_slope
        return seq_df

    def sigmoidal_fitness(self, seq_df: pd.DataFrame):
        r"""Combines methods to get the antigen bound, Tfh help, and fitness
        from the KD using a sigmoidal model."""
        seq_df["frac_antigen_bound"] = self.frac_antigen_bound(seq_df["KD"])
        for idx, row in seq_df.iterrows():
            antigen_bound = row.frac_antigen_bound
            Tfh_help = self.maximum_Tfh / (
                1
                + exp(
                    -1
                    * self.curve_steepness
                    * (antigen_bound - self.midpoint_antigen_bound)
                )
            )
            seq_df.loc[idx, "fitness"] = Tfh_help
        return seq_df

    def fitness(
        self,
        seq_list: list[str] = None,
        KD_calculator: Callable[list[str], list[float]] = None,
    ):
        r"""Produces the fitness of a series of sequences given KD values.

        Args:
            seq_list: list of DNA sequences
            KD_calculator: method that produces a KD value for each sequence in the list
        Returns:
            list of fitness values based on function determined by `mapping_type`
        """
        seq_df = self.fitness_df(seq_list, KD_calculator)
        return seq_df["fitness"]

    def fitness_df(
        self,
        seq_list: list[str] = None,
        KD_calculator: Callable[list[str], list[float]] = None,
    ):
        r"""Produces a dataframe including the fitness of a series of sequences given KD values.

        Args:
            seq_list: list of DNA sequences
            KD_calculator: method that produces a KD value for each sequence in the list/file
        Returns:
            DataFrame with columns `frac_antigen_bound`, `KD`, and `fitness`,
            with fitness values based on function determined by `mapping_type`
        """
        KD_values = KD_calculator(seq_list)
        seq_df = pd.DataFrame({"seq": seq_list, "KD": KD_values})
        return self.fitness_func(seq_df)

    def normalize_fitness(self, seq_df: pd.DataFrame):
        """Normalize fitness from a dataframe with a fitness column."""
        sum_fitness = seq_df["fitness"].sum()
        seq_df["normalized_fitness"] = (seq_df["fitness"]) / (sum_fitness)
        return seq_df["normalized_fitness"]

    def map_cell_divisions(self, seq_df: pd.DataFrame, slope: float = 1):
        """Map fitness linearly to the number of cell divisions using slope."""
        seq_df["cell_divisions"] = seq_df["normalized_fitness"] * slope
        return seq_df["cell_divisions"]
