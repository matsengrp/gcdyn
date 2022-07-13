from gcdyn.fitness import Fitness
from gcdyn.replay import ReplayPhenotype
from Bio import SeqIO
import pytest


@pytest.fixture
def seq_list():
    seqs = [
        "GAGGTGCAGCTTCAGGAGTCAGGACCTAACCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTGTCACAGGCGACTCCATCACCAGTGGTTACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGGTACATAAGCTACAGTGGTAGCACTTACTACAATCCATCTCTCAAAAGTCGAATCTCCATCACTCGAGACACATCCAAGAATCAGTACTACCTGAAGTTGAATTCTGTGACTACTGAGGACACAGCCTCATACTACTGTGGAAGGGACTTCGATGTCTGGGGCGCAGGGACCACGGTCATCGTCTCCTCAGACATTGTGATGACTCAGTCTCAAAAATTCATGTCCACATCAGTAGGAGACAGGGTCAGCGTCACCTGCAAGGCCAGTCAGAATGTGGGTACTAATGTAGCCTGGTATCAACAGAAACCAGGGCAATCTCCTAAAGCACTGATTTACTCGGCATCCTACAGGTACAGTGGAGTCCCTGATCGCTTCACAGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAATGTGCAGTCTGAAGACTTGGCAGAGTATTTCTGTCACCAATATAGCAGCTATCCTCTCACGTTCGGCTCGGGGACTAAGCTAGAAATAAAA",
        "GAGGTGCAGCTTCAGGAGTCAGGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTGTCACTGGCGACTCCATCACCAGTGGTTACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGGTACATAAGCTACAGTGGTCGCACTTACTACAATCCATCTCTCATAAGTCGAATCTCCATCACTCGAGACACATCCAAGAACCAGTACTACCTGCAGTTGAATTCTGTGACTACTGAGGACACAGCCACATATTACTGTGGAAGGGACTTCGATGTCTGGGGCGCAGGGACCACGGTCACCGTCTCCTCAGACATTGTGATGACTCAGTCTCAAAAATTCATGTCCACATCAGTAGGAGACAGGGTCAGCGTCACCTGCAAGGCCAGTCAGAATGTGGGTACTAATGTAGCCTGGTATCAACAGAAACCAGGGCAATCTCCTAAAGCACTGATTTACTCGGCATCCTACAGGTACAGTGGAGTCCCTGATCGCTTCACAGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAATGTGCAGTCTGAAGACTTGGCAGAGTATTTCTGTCAGCAATATAACAGCTATCCTCTCACGTTCGGCTCGGGGACTAAGCTAGAAATAAAA",
        "GAGGTGCAGCTTCAGGAGTCAGGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTGTCACTGGCGACTCCATCACCAGTGGTTACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGGTACATAAGCTACAGTGGTAACACTTACTACAATCCATCTCTCAAAAGTCGAATCTCCATCACTCGAGACACATCCAAGAACCAGTACTACCTGCAGTTGAATTCTGTGACTACTGAGGACACAGCCACATATTACTGTGCAAGGGACTTCGATGTCTGGGGCGCAGGGACCACGGTCACCGTCTCCTCAGACATTGTGATGACTCAGTCTCAAAAATTCATGTCCACATCAGTAGGAGACAGGGTCAGCGTCACCTGCAAGGCCAGTCAGAATGTGGGTACTAATGTAGCCTGGTATCAACAGAAACCAGGACAATCTCCTAGATCACTGATTTACTCGGCATCCTACAGGTACAGTGGAGTCCCTGATCGCTTCACAGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAATGTGCAGTCTGAAGACTTGGCCGAGTATTTCTGTCATCAATATAGCAGCTATCCTCTCACGTTCGGCTCGGGGACTAAGCTAGAAATAAAA",
    ]

    return seqs


@pytest.fixture
def seq_list_big():
    seqs = [
        str(seq_record.seq)
        for seq_record in SeqIO.parse(
            "notebooks/gcreplay_samples/gctree_PR1.2-5-LP-78-GC.fasta", "fasta"
        )
        if seq_record.id != "naive"
    ]
    return seqs


@pytest.fixture
def fasta_seq_path():
    path = "notebooks/sample.fasta"
    return path


@pytest.fixture
def replay_phenotype():
    replay_phenotype = ReplayPhenotype(
        1,
        1,
        336,
        "https://raw.githubusercontent.com/jbloomlab/Ab-CGGnaive_DMS/main/data/CGGnaive_sites.csv",
        "notebooks/Linear.model",
        ["delta_log10_KD", "expression"],
        -10.43,
    )
    return replay_phenotype


def test_fitness(seq_list, replay_phenotype):
    fit = Fitness(Fitness.linear_fitness)
    linear_fitness_df = fit.normalized_fitness_df(
        seq_list=seq_list, calculate_KD=replay_phenotype.calculate_KD
    )
    print(linear_fitness_df)
    assert all(fitness > 0 for fitness in linear_fitness_df["t_cell_help"])


def test_normalized_fitness(seq_list, replay_phenotype):
    fit = Fitness(Fitness.sigmoidal_fitness)
    sig_fitness_df = fit.normalized_fitness_df(
        seq_list=seq_list, calculate_KD=replay_phenotype.calculate_KD
    )
    print(sig_fitness_df)
    assert all(1 > fitness > 0 for fitness in sig_fitness_df["normalized_t_cell_help"])


def test_uniform_fitness(seq_list, replay_phenotype):
    fit = Fitness(Fitness.uniform_fitness)
    uniform_fitness_df = fit.normalized_fitness_df(
        seq_list=seq_list, calculate_KD=replay_phenotype.calculate_KD
    )
    print(uniform_fitness_df)
    assert all(
        1 > fitness > 0 for fitness in uniform_fitness_df["normalized_t_cell_help"]
    )


def test_normalized_fitness_big(seq_list_big, replay_phenotype):
    fit = Fitness(Fitness.sigmoidal_fitness)
    sig_fitness_df = fit.normalized_fitness_df(
        seq_list=seq_list_big, calculate_KD=replay_phenotype.calculate_KD
    )
    assert all(1 > fitness > 0 for fitness in sig_fitness_df["normalized_t_cell_help"])
