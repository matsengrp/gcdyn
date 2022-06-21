import pandas as pd, torch, numpy
from Bio import SeqIO
from Bio.Seq import Seq
from plotnine import ggplot, geom_histogram, aes, facet_wrap, ggtitle, xlim, ylim

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

igh_frame = 1
igk_frame = 1
igk_idx = 336
naive_sites_path = "https://raw.githubusercontent.com/jbloomlab/Ab-CGGnaive_DMS/main/data/CGGnaive_sites.csv"

# collect the correct sites df from tylers repo
pos_df = pd.read_csv(
        naive_sites_path,
        dtype=dict(site=pd.Int16Dtype()),
        index_col="site_scFv",
    )

# evaluate sequences using torchdms model and return the evaluation as a pandas dataframe
def evaluate(torchdms_model, seqs: list[str], phenotype_names: list[str]):
    aa_seq_one_hot = torch.stack([torchdms_model.seq_to_binary(seq) for seq in seqs])
    try:
        labeled_evaluation = pandas.DataFrame(torchdms_model(aa_seq_one_hot).detach().numpy(), columns=phenotype_names)
    except ValueError:
        print("Incorrect number of column labels for phenotype data")
    return labeled_evaluation
