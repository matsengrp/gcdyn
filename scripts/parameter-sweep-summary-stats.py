r"""Produce summary statistics for simulated sequences."""
import argparse
from typing import Callable, List

import numpy as np
from ete3 import TreeNode
from numpy.random import default_rng

import gcdyn.cycles as cycles
from gcdyn.phenotype import DMSPhenotype
from Bio import SeqIO

parser = argparse.ArgumentParser()
parser.add_argument("--naivefasta", type=str, help="fasta with naive BCR sequence")
parser.add_argument("--out", type=str, help="csv outfile prefix")
parser.add_argument(
    "--nsteps",
    type=int,
    help="number of times to simulate tree between minval and maxval",
)
parser.add_argument(
    "--numcycles", type=int, help="number of DZ/LZ cycles to simulate", default=5
)
parser.add_argument(
    "--nsamples",
    type=int,
    help="number of resultant sequences to randomly sample for statistics (samples all by default)",
    default=None,
)
parser.add_argument("--parameter", type=str, help="parameter with varying value")
parser.add_argument(
    "--minval", type=float, help="minimum value for parameter (if applicable)"
)
parser.add_argument(
    "--maxval", type=float, help="maximum value for parameter (if applicable)"
)

args = parser.parse_args()


def simulate_cycle_seqs(
    naive_seq: str,
    N0: int = 10,
    Nmax: int = None,
    mutator: cycles.Mutator = cycles.FivemerMutator(
        mutability_csv="notebooks/MK_RS5NF_mutability.csv",
        substitution_csv="notebooks/MK_RS5NF_substitution.csv",
    ),
    selector: cycles.Selector = None,
    proliferator: Callable[
        [TreeNode, float, np.random.Generator], None
    ] = cycles.cell_div_balanced_proliferator,
    total_t_cell_help: float = 1000,
    concentration_antigen: float = 1e-10,
    max_help: float = 6,
    antigen_frac_limit=0.2,
    model_path: str = "notebooks/tdms-linear.model",
    sigmoid_growth_rate: float = 10,
    sigmoid_mid_competency: float = 0.5,
):
    """
    Simulate a GC from the naive sequence for a number of cycles
    Args:
        naive_seq: naive sequence
        N0: number of naive sequences in GC initially
        Nmax: maximum number of sequences after any cycle
        mutator: Mutator to act after proliferation
        selector: Selector to determine fitness (cell divisions) of each cell in GC
        proliferator: method to create cell divisions based on fitness values
        total_t_cell_help: total amount of T cell help to be distributed (for three step selector)
        concentration_antigen: concentration of antigen to use to determine antigen bound
        max_help: maximum amount of T cell help that may be attained by each cell
        antigen_frac_limit: the inclusive lower limit of antigen bound to have any division (for three step selector)
        model_path: path to model to determine KD
        sigmoid_growth_rate: logistic growth rate of signal in selector
        sigmoid_mid_competency: value of input competency to set as midpoint in selector

    Returns:
        genotypes: list of resultant BCR sequences from tree
    """
    if selector is None:
        selector = cycles.ThreeStepSelector(
            concentration_antigen=concentration_antigen,
            total_t_cell_help=total_t_cell_help,
            model_path=model_path,
            max_help=max_help,
            antigen_frac_limit=antigen_frac_limit,
            sigmoid_mid_competency=sigmoid_mid_competency,
            sigmoid_growth_rate=sigmoid_growth_rate,
            tdms_phenotypes=["delta_log10_KD", "delta_expression", "delta_psr"],
        )
    gc = cycles.GC(
        naive_seq,
        proliferator,
        mutator,
        selector,
        N0=N0,
        Nmax=Nmax,
    )
    gc.simulate(args.numcycles, enforce_timescale=True)
    sequences = []
    for leaf in gc.alive_leaves:
        sequences.append(leaf.sequence)
    return sequences


def simulated_summary_stats(
    naive_seq: str,
    sequences: List[str],
    nsamples: int = None,
):
    """
    Sample from simulated trees and calculate statistics.
    Args:
        naive_seq: original sequence to calculate hamming distance from descendants
        sequences: list of all alive B cell BCR sequences
        nsamples: number of genotypes to sample


    Returns:
        summary_stats: tuple of row values for output:
        # of unique sequences, total # of sequences, # of sampled sequences, mean abundance mean distance,
        median affinity, mean affinity, 1st quartile, 3rd quartile, min affinity, max affinity,
        standard deviation of affinity
    """
    distances = []
    abundances = []
    affinities = []
    phenotype = DMSPhenotype(
        1,
        1,
        336,
        "https://raw.githubusercontent.com/jbloomlab/Ab-CGGnaive_DMS/main/data/CGGnaive_sites.csv",
        "notebooks/tdms-linear.model",
        ["delta_log10_KD", "delta_expression", "delta_psr"],
        -10.43,
    )
    kd_vals = phenotype.calculate_KD(sequences)
    seq_to_kd = dict(zip(sequences, kd_vals))
    rng = default_rng()
    if nsamples is None or nsamples > len(sequences):
        nsamples = len(sequences)
    genotypes = {}
    for sequence in rng.choice(
        sequences,
        size=nsamples,
        replace=False,
    ):
        if sequence in genotypes:
            genotypes[sequence] += 1
        else:
            genotypes[sequence] = 1
    for genotype_sequence in genotypes.keys():
        abundances.append(genotypes[genotype_sequence])
        hamming_dist = sum(
            naive_nt != sim_nt for naive_nt, sim_nt in zip(naive_seq, genotype_sequence)
        )
        distances.append(hamming_dist)
        affinities.append(seq_to_kd[genotype_sequence])
    return (
        len(np.unique(sequences)),
        len(genotypes.keys()),
        nsamples,
        np.median(abundances),
        np.mean(abundances),
        np.mean(distances),
        np.median(affinities),
        np.mean(affinities),
        np.quantile(affinities, 0.25),
        np.quantile(affinities, 0.75),
        np.min(affinities),
        np.max(affinities),
        np.std(affinities),
    )


stats_dict = {}

seqs = [
    str(seq_record.seq)
    for seq_record in SeqIO.parse(args.naivefasta, "fasta")
    if seq_record.id == "naive"
]
assert len(seqs) == 1
naive_sequence = seqs[0]

if args.minval is None and args.maxval is None:
    param_vals = [1]
else:
    param_vals = np.linspace(args.minval, args.maxval, args.nsteps)
parameter = args.parameter
for param_val in param_vals:
    if parameter == "N0":
        genotypes = simulate_cycle_seqs(naive_sequence, N0=int(param_val))
    elif parameter == "Nmax":
        genotypes = simulate_cycle_seqs(naive_sequence, Nmax=int(param_val))
    elif parameter == "total_t_cell_help":
        genotypes = simulate_cycle_seqs(
            naive_sequence, total_t_cell_help=int(param_val)
        )
    elif parameter == "concentration_antigen":
        genotypes = simulate_cycle_seqs(naive_sequence, concentration_antigen=param_val)
    elif parameter == "max_help":
        genotypes = simulate_cycle_seqs(naive_sequence, max_help=param_val)
    elif parameter == "sigmoid_growth_rate":
        genotypes = simulate_cycle_seqs(naive_sequence, sigmoid_growth_rate=param_val)
    elif parameter == "sigmoid_mid_competency":
        genotypes = simulate_cycle_seqs(
            naive_sequence, sigmoid_mid_competency=param_val
        )
    elif parameter == "antigen_frac_limit":
        genotypes = simulate_cycle_seqs(naive_sequence, antigen_frac_limit=param_val)
    elif parameter == "divisions_help_slope":
        genotypes = simulate_cycle_seqs(naive_sequence, divisions_help_slope=param_val)
    else:
        raise (ValueError(f"invalid parameter choice: {args.parameter}"))
    stats_dict[param_val] = simulated_summary_stats(
        naive_sequence, genotypes, nsamples=args.nsamples
    )


# write csv
with open(args.out, "w") as fh:
    print(
        "parameter,parameter value,total number of genotypes,number of sampled genotypes,number of samples,"
        "median abundance,mean abundance,mean number of substitutions,median affinity,mean affinity, 25th percentile,"
        "75th percentile,min affinity,max affinity, affinity std",
        file=fh,
    )
    for param_val, stats in stats_dict.items():
        stat_str = ",".join(str(stat) for stat in stats)
        print(
            f"{args.parameter},{param_val},{stat_str}",
            file=fh,
        )
