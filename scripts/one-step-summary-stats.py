r"""Produce summary statistics for simulated sequences."""
import argparse
import os
from typing import Callable, List
import numpy as np
from Bio import SeqIO
from ete3 import TreeNode
from numpy.random import default_rng
import gcdyn.cycles as cycles
from gcdyn.phenotype import DMSPhenotype

parser = argparse.ArgumentParser()
parser.add_argument("--fastadir", type=str, help="directory with fasta files to read")
parser.add_argument("--out", type=str, help="csv outfile prefix")
parser.add_argument(
    "--nsteps",
    type=int,
    help="number of times to simulate tree between minval and maxval",
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
    seqs: List[str],
    mutator: cycles.Mutator = None,
    selector: cycles.Selector = None,
    proliferator: Callable[
        [TreeNode, float, np.random.Generator], None
    ] = cycles.cell_div_balanced_proliferator,
    total_t_cell_help: float = 550,
    concentration_antigen: float = 1e-10,
    max_help: float = 6,
    antigen_frac_limit=0.2,
    model_path: str = "notebooks/tdms-linear.model",
    sigmoid_growth_rate: float = 10,
    sigmoid_mid_competency: float = 0.5,
):
    """
    Simulate a GC from a list of sequences for one cycle
    Args:
        seqs: set of starting seqs
        mutator: Mutator to act after proliferation
        selector: Selector to determine fitness (cell divisions) of each cell in GC
        proliferator: method to create cell divisions based on fitness values
        total_t_cell_help: total amount of T cell help to be distributed (for three step selector)
        concentration_antigen: concentration of antigen to use to determine antigen bound
        max_help: maximum amount of T cell help that may be attained by each cell
        antigen_frac_limit: the inclusive lower limit of antigen bound to have any division (for three step selector)
        model_path: path to model to determine KD

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
    if mutator is None:
        mutator = cycles.FivemerMutator(
            mutability_csv="notebooks/MK_RS5NF_mutability.csv",
            substitution_csv="notebooks/MK_RS5NF_substitution.csv",
        )
    if proliferator is None:
        proliferator = cycles.cell_div_balanced_proliferator
    fitnesses = selector.select(seqs)
    alive_leaves = []
    for seq in seqs:
        leaf = TreeNode()
        leaf.dist = 0
        leaf.sequence = seq
        alive_leaves.append(leaf)
    descendant_seqs = []
    for leaf, args in zip(alive_leaves, fitnesses):
        proliferator(leaf, *args)
        for node in leaf.iter_descendants():
            node.sequence = mutator.mutate(node.up.sequence, node.dist)
            if node.is_leaf() and node != leaf:
                descendant_seqs.append(node.sequence)
    return descendant_seqs


def simulated_summary_stats(
    naive_seq: str, sequences: List[str], nsamples: int = None, param_val: float = None
):
    """
    Sample from simulated trees and calculate statistics.
    Args:
        seq: original sequence to calculate hamming distance from descendants
        sequences: list of all alive B cell BCR sequences
        nsamples: number of genotypes to sample
        param_val: parameter value to record in dictionary

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
        "notebooks/Linear.model",
        ["delta_log10_KD", "delta_expression"],
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
        param_val,
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


def sweep_param_vals(
    naive_seq: str,
    seqs: List[str],
    parameter: str,
    minval: float = None,
    maxval: float = None,
    nsteps: int = 1,
    nsamples: int = None,
):
    if minval is None and maxval is None:
        param_vals = [1]
    else:
        param_vals = np.linspace(minval, maxval, nsteps)
    stats = []
    for param_val in param_vals:
        if parameter == "total_t_cell_help":
            descendant_seqs = simulate_cycle_seqs(
                seqs, total_t_cell_help=int(param_val)
            )
        elif parameter == "concentration_antigen":
            descendant_seqs = simulate_cycle_seqs(seqs, concentration_antigen=param_val)
        elif parameter == "max_help":
            descendant_seqs = simulate_cycle_seqs(seqs, max_help=param_val)
        elif parameter == "sigmoid_growth_rate":
            descendant_seqs = simulate_cycle_seqs(seqs, sigmoid_growth_rate=param_val)
        elif parameter == "sigmoid_mid_competency":
            descendant_seqs = simulate_cycle_seqs(
                seqs, sigmoid_mid_competency=param_val
            )
        elif parameter == "antigen_frac_limit":
            descendant_seqs = simulate_cycle_seqs(seqs, antigen_frac_limit=param_val)
        elif parameter == "divisions_help_slope":
            descendant_seqs = simulate_cycle_seqs(seqs, divisions_help_slope=param_val)
        else:
            raise (ValueError(f"invalid parameter choice: {args.parameter}"))
        stats.append(
            simulated_summary_stats(naive_seq, descendant_seqs, nsamples, param_val)
        )
    return stats


# MAIN #

stats_dict = {}
naive_seq = "GAGGTGCAGCTTCAGGAGTCAGGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTGTCACTGGCGACTCCATCACCAGTGGTTACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGGTACATAAGCTACAGTGGTAGCACTTACTACAATCCATCTCTCAAAAGTCGAATCTCCATCACTCGAGACACATCCAAGAACCAGTACTACCTGCAGTTGAATTCTGTGACTACTGAGGACACAGCCACATATTACTGTGCAAGGGACTTCGATGTCTGGGGCGCAGGGACCACGGTCACCGTCTCCTCAGACATTGTGATGACTCAGTCTCAAAAATTCATGTCCACATCAGTAGGAGACAGGGTCAGCGTCACCTGCAAGGCCAGTCAGAATGTGGGTACTAATGTAGCCTGGTATCAACAGAAACCAGGGCAATCTCCTAAAGCACTGATTTACTCGGCATCCTACAGGTACAGTGGAGTCCCTGATCGCTTCACAGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAATGTGCAGTCTGAAGACTTGGCAGAGTATTTCTGTCAGCAATATAACAGCTATCCTCTCACGTTCGGCTCGGGGACTAAGCTAGAAATAAAA"

for file in os.listdir(args.fastadir):
    if file.endswith(".fasta"):
        seqs = [
            str(seq_record.seq)
            for seq_record in SeqIO.parse(f"{args.fastadir}/{file}", "fasta")
            if seq_record.id != "naive"
        ]
        stats_dict[file] = sweep_param_vals(
            naive_seq=naive_seq,
            seqs=seqs,
            parameter=args.parameter,
            minval=args.minval,
            maxval=args.maxval,
            nsteps=args.nsteps,
            nsamples=args.nsamples,
        )
# write csv
with open(args.out, "w") as fh:
    print(
        "filename,parameter,parameter value,total number of genotypes,number of sampled genotypes,number of samples,"
        "median abundance,mean abundance,mean number of substitutions,median affinity,mean affinity, 25th percentile,"
        "75th percentile,min affinity,max affinity, affinity std",
        file=fh,
    )
    for filename, stats in stats_dict.items():
        for stat in stats:
            stat_str = ",".join(str(val) for val in stat)
            print(
                f"{filename},{args.parameter},{stat_str}",
                file=fh,
            )
