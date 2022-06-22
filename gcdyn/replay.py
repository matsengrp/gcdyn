import pandas as pd, torch, numpy
from Bio import SeqIO
from Bio.Seq import Seq
from plotnine import ggplot, geom_histogram, aes, facet_wrap, ggtitle, xlim, ylim, geom_point
from math import exp

igh_frame = 1
igk_frame = 1
igk_idx = 336
naive_sites_path = "https://raw.githubusercontent.com/jbloomlab/Ab-CGGnaive_DMS/main/data/CGGnaive_sites.csv"
model_path = "Linear.model"
model = torch.load(model_path)
tdms_phenotypes = ['delta_log10_KD', 'expression']
log10_naive_KD = -10.43
α = 10
β = .5
k = 4
c = 1

# From gcreplay-tools (https://github.com/matsengrp/gcreplay/blob/main/nextflow/bin/gcreplay-tools.py):

def fasta_to_df(f):
    """simply convert a fasta to dataframe"""
    ids, seqs = [], []
    with open(f) as fasta_file:
        for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
            if str(seq_record.id) != 'naive':
                ids.append(seq_record.id)
                seqs.append(str(seq_record.seq))
    return pd.DataFrame({"id":ids, "seq":seqs})


def aa(sequence, frame):
    """Amino acid translation of nucleotide sequence in frame 1, 2, or 3."""
    return Seq(
        sequence[(frame - 1): (frame - 1
                               + (3 * ((len(sequence) - (frame - 1)) // 3)))]
    ).translate()

def read_sites_file(naive_sites_path: str):
    """Read the sites file from csv"""
    # collect the correct sites df from tylers repo
    pos_df = pd.read_csv(
            naive_sites_path,
            dtype=dict(site=pd.Int16Dtype()),
            index_col="site_scFv",
        )
    return pos_df


def seq_df_tdms(fasta_path: str, igh_frame: int, igk_frame: int, igk_idx: int):
    """Make amino acid sequence predictions using chain information and return as a dataframe"""
    # load the seqs from a fasta
    seqs_df = fasta_to_df(fasta_path)

    # make a prediction for each of the observed sequences
    for idx, row in seqs_df.iterrows():

        # translate heavy and light chains
        igh_aa = aa(row.seq[:igk_idx], igh_frame)
        igk_aa = aa(row.seq[igk_idx:], igk_frame)

        # Make the aa seq for tdms
        aa_tdms = pos_df.amino_acid.copy()
        aa_tdms.iloc[pos_df.chain == "H"] = igh_aa
        # note: replay light chains are shorter than dms seq by one aa
        aa_tdms.iloc[(pos_df.chain == "L") & (pos_df.index < pos_df.index[-1])] = igk_aa
        aa_tdms_seq = "".join(aa_tdms)
        seqs_df.loc[idx, "aa_sequence"] = aa_tdms_seq

    return seqs_df


# phenotype evaluation:

def evaluate(torchdms_model, seqs: list[str], phenotype_names: list[str]):
    """Evaluate sequences using torchdms model and return the evaluation as a pandas dataframe"""
    aa_seq_one_hot = torch.stack([torchdms_model.seq_to_binary(seq) for seq in seqs])
    try:
        labeled_evaluation = pd.DataFrame(torchdms_model(aa_seq_one_hot).detach().numpy(), columns=phenotype_names)
    except ValueError:
        print("Incorrect number of column labels for phenotype data")
    return labeled_evaluation
