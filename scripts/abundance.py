"""
Calculate the sequence abundance frequency for each input fasta and then write it as a CSV.
"""

import collections
import os
import sys
import pandas as pd

from Bio import SeqIO
from Bio.SeqUtils.CheckSum import seguid

abundances = {}

for fasta_path in sys.argv[1:]:

    records = SeqIO.parse(fasta_path, "fasta")

    # This dictionary will map sequence checksum to the list of squence ids that have that
    # sequence checksum.
    ids_by_checksum = collections.defaultdict(list)
    sequence_count = 0
    for record in records:
        sequence_count = sequence_count + 1
        ids_by_checksum[seguid(record.seq)].append(record.id)

    abundance_distribution = collections.defaultdict(int)

    for id_list in ids_by_checksum.values():
        id_count = len(id_list)
        abundance_distribution[id_count] = abundance_distribution[id_count] + 1

    assert sequence_count == sum(k * v for k, v in abundance_distribution.items())

    base, _ = os.path.splitext(fasta_path)
    base = os.path.basename(base)
    abundances[base] = pd.Series(
        abundance_distribution.values(), index=abundance_distribution.keys()
    )

to_write = pd.DataFrame(abundances).fillna(0).astype(int)
to_write.to_csv("abundances.csv")
