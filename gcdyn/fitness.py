r"""Uses phenotype to determine fitness"""
import gcdyn.replay as replay
from math import exp


class Fitness:
    r"""Class to determine fitness from phenotype for a collection of sequences

    Args:
        fasta_path: path to a list of DNA sequences to use for phenotype prediction
        DNA_seq_list: list of DNA sequences to use for phenotype prediction (overridden if fasta provided)
    """

    def __init__(self, fasta_path: str = None, DNA_seq_list: list[str] = None):
        if fasta_path is not None:
            seqs_df = replay.seq_df_tdms(
                replay.igh_frame,
                replay.igk_frame,
                replay.igk_idx,
                fasta_path=fasta_path,
            )
        elif DNA_seq_list is not None:
            seqs_df = replay.seq_df_tdms(
                replay.igh_frame,
                replay.igk_frame,
                replay.igk_idx,
                seq_list=DNA_seq_list,
            )
        else:
            raise Exception(
                "Must define a path to DNA sequences or a list of sequences"
            )
        delta_log_KDs = replay.evaluate(
            replay.model, seqs_df["aa_sequence"], replay.tdms_phenotypes
        )
        self.all_info_df = seqs_df.merge(
            delta_log_KDs, left_index=True, right_index=True
        )

    def frac_antigen_bound(self, log10_naive_KD: float, concentration_antigen: float):
        """"""
        delta_log_KDs = self.all_info_df["delta_log10_KD"]
        thetas = []
        for delta_log_KD in delta_log_KDs:
            log10_KD = delta_log_KD + log10_naive_KD
            KD = 10**log10_KD
            # Hill equation with n = 1:
            theta = concentration_antigen / (KD + concentration_antigen)
            thetas.append(theta)
        self.all_info_df["frac_antigen_bound"] = thetas

    def linear_fitness(
        self, slope: float, log10_naive_KD: float, concentration_antigen: float
    ):
        """Combines methods to get the antigen bound, Tfh help, and fitness
        from the KD using a linear model."""
        self.frac_antigen_bound(log10_naive_KD, concentration_antigen)
        self.all_info_df["fitness"] = self.all_info_df["frac_antigen_bound"] * slope
        return self.all_info_df["fitness"]

    def sigmoidal_fitness(
        self,
        maximum_Tfh: float,
        curve_steepness: float,
        midpoint_antigen_bound: float,
        log10_naive_KD: float,
        concentration_antigen: float,
    ):
        """Combines methods to get the antigen bound, Tfh help, and fitness
        from the KD using a sigmoidal model."""
        self.frac_antigen_bound(log10_naive_KD, concentration_antigen)
        for idx, row in self.all_info_df.iterrows():
            antigen_bound = row.frac_antigen_bound
            Tfh_help = maximum_Tfh / (
                1 + exp(-1 * curve_steepness * (antigen_bound - midpoint_antigen_bound))
            )
            self.all_info_df.loc[idx, "fitness"] = Tfh_help
        return self.all_info_df["fitness"]

    def fitness(
        self,
        mapping_type: str = "linear",
        log10_naive_KD: float = -10.43,
        concentration_antigen: float = 10 ** (-9),
        slope: float = 1,
        maximum_Tfh: float = 4,
        curve_steepness: float = 10,
        midpoint_antigen_bound: float = 0.5,
    ):
        r"""Maps evaluation to unnormalized fitness
        Args:
            mapping_type: type of mapping function (defaults to linear)
            log10_naive_KD: log of naive KD used to infer absolute KD
            concentration_antigen: concentration used to infer antigen bound
            slope: slope for linear mapping between antigen bound and fitness
            maximum_Tfh: maximum Tfh help for sigmoidal mapping to fitness
            curve_steepness: logistic growth rate for sigmoidal mapping to fitness
            midpoint_antigen_bound: midpoint of antigen bound for sigmoidal mapping to fitness
        """
        if mapping_type == "linear":
            return self.linear_fitness(slope, log10_naive_KD, concentration_antigen)
        elif mapping_type == "sigmoid":
            return self.sigmoidal_fitness(
                maximum_Tfh,
                curve_steepness,
                midpoint_antigen_bound,
                log10_naive_KD,
                concentration_antigen,
            )
        else:
            raise Exception("Only linear and sigmoid are acceptable mapping types")

    def normalize_fitness(self):
        """Normalize fitness from a dataframe with a fitness column."""
        sum_fitness = self.all_info_df["fitness"].sum()
        self.all_info_df["normalized_fitness"] = (self.all_info_df["fitness"]) / (
            sum_fitness
        )
        return self.all_info_df["normalized_fitness"]

    def map_cell_divisions(self, slope: float = 1):
        """Map fitness linearly to the number of cell divisions using slope."""
        self.all_info_df["cell_divisions"] = (
            self.all_info_df["normalized_fitness"] * slope
        )
        return self.all_info_df["cell_divisions"]
