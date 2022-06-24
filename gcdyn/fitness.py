r"""Uses phenotype to determine fitness"""
from __future__ import annotations
from gcdyn.replay import ReplayPhenotype, fasta_to_df, aa, read_sites_file
from math import exp


class Fitness:
    r"""Class to determine fitness from phenotype for a collection of sequences

    Args:
        mapping_type: type of mapping function (defaults to linear)
        log10_naive_KD: log of naive KD used to infer absolute KD
        concentration_antigen: concentration used to infer antigen bound
        slope: slope for linear mapping between antigen bound and fitness
        maximum_Tfh: maximum Tfh help for sigmoidal mapping to fitness
        curve_steepness: logistic growth rate for sigmoidal mapping to fitness
        midpoint_antigen_bound: midpoint of antigen bound for sigmoidal mapping to fitness
        fasta_path: path to a list of DNA sequences to use for phenotype prediction
        DNA_seq_list: list of DNA sequences to use for phenotype prediction (overridden if fasta provided)
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
        self.mapping_type = mapping_type
        self.linfit_slope = linfit_slope
        self.maximum_Tfh = maximum_Tfh
        self.curve_steepness = curve_steepness
        self.midpoint_antigen_bound = midpoint_antigen_bound
        self.concentration_antigen = concentration_antigen

    def frac_antigen_bound(self, KD_values: list[float]):
        r"""Determine the fraction of antigen bound using the Hill equation."""
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
                1 + exp(-1 * self.curve_steepness * (antigen_bound - self.midpoint_antigen_bound))
            )
            seq_df.loc[idx, "fitness"] = Tfh_help
        return seq_df

    def fitness(self, fasta_path: str = None, seq_list: list[str] = None):
        seq_df = fitness_df(fasta_path, seq_list)
        return seq_df["fitness"]

    def fitness_df(self, fasta_path: str = None, seq_list: list[str] = None):
        r"""Maps evaluation to unnormalized fitness"""
        replay_phenotype = ReplayPhenotype(
            1,
            1,
            336,
            "https://raw.githubusercontent.com/jbloomlab/Ab-CGGnaive_DMS/main/data/CGGnaive_sites.csv",
            "notebooks/Linear.model",
            ["delta_log10_KD", "expression"],
            -10.43,
        )
        seq_df = replay_phenotype.calculate_KD_df(fasta_path, seq_list)
        if self.mapping_type == "linear":
            return self.linear_fitness(seq_df)
        elif self.mapping_type == "sigmoid":
            return self.sigmoidal_fitness(seq_df)
        else:
            raise Exception("Only linear and sigmoid are acceptable mapping types")

    def normalize_fitness(self, seq_df: pd.DataFrame):
        """Normalize fitness from a dataframe with a fitness column."""
        sum_fitness = seq_df["fitness"].sum()
        seq_df["normalized_fitness"] = (seq_df["fitness"]) / (
            sum_fitness
        )
        return seq_df["normalized_fitness"]

    def map_cell_divisions(self, seq_df: pd.DataFrame, slope: float = 1):
        """Map fitness linearly to the number of cell divisions using slope."""
        seq_df["cell_divisions"] = (
            seq_df["normalized_fitness"] * slope
        )
        return seq_df["cell_divisions"]
