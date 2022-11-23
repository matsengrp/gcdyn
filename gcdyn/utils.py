r"""
Utility functions
^^^^^^^^^^^^^^^^^
"""


def simple_fivemer_contexts(sequence):
    """Decompose a sequence into a list of its 5mer contexts."""
    return [sequence[(i - 2) : (i + 3)] for i in range(2, len(sequence) - 2)]


def padded_fivemer_contexts_of_paired_sequences(sequence, chain_2_start_idx):
    """Given a pair of sequences with two chains, split them apart at the given index,
    then generate all of the 5mer contexts for each of the chains, padding
    appropriately.
    """
    chain_1_seq = "NN" + sequence[:chain_2_start_idx] + "NN"
    chain_2_seq = "NN" + sequence[chain_2_start_idx:] + "NN"
    return simple_fivemer_contexts(chain_1_seq) + simple_fivemer_contexts(chain_2_seq)


def write_leaf_sequences_to_fasta(tree, file_path, naive=None):
    """Write the sequences at the leaves of a tree to a FASTA file, potentially
    including a naive sequence."""
    sequence_dict = {leaf.name: leaf.sequence for leaf in tree.iter_leaves()}
    if naive is not None:
        sequence_dict["naive"] = naive
    with open(file_path, "w") as fp:
        for name, sequence in sequence_dict.items():
            fp.write(f">{name}\n")
            fp.write(f"{sequence}\n")
