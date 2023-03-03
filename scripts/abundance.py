"""
Calculate the sequence abundance frequency for each input fasta and then write it as a CSV.
"""

import collections
import os
import sys
import pandas as pd
import argparse
import numpy

from Bio import SeqIO
from Bio.SeqUtils.CheckSum import seguid

ustr = """
python abundance.py <input fasta files> [--outdir <outdir>]
"""
parser = argparse.ArgumentParser(usage=ustr)
parser.add_argument('infiles', nargs='+')
parser.add_argument('--outdir', default=os.getcwd())
parser.add_argument('--min-seqs', type=int, help='only keep GCs that have this many seqs')
parser.add_argument('--max-seqs', type=int, help='downsample to this many seqs any GCs that have more than this')
args = parser.parse_args()

n_too_small, init_sizes = 0, []
abundances = {}

for fasta_path in args.infiles:
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if args.min_seqs is not None and len(records) < args.min_seqs:
        n_too_small += 1
        continue
    if args.max_seqs is not None and len(records) > args.max_seqs:
        init_sizes.append(len(records))
        i_to_keep = numpy.random.choice(list(range(len(records))), size=args.max_seqs)  # i_to_keep is because numpy.random.choice barfs on <records> because each record handles the [] operator (@$$*$~$~!!)
        records = [records[i] for i in i_to_keep]

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

if n_too_small > 0:
    print('    skipped %d files with fewer than %d seqs' % (n_too_small, args.min_seqs))
if len(init_sizes) > 0:
    print('    downsampled %d samples to %d from initial sizes: %s' % (len(init_sizes), args.max_seqs, ' '.join(str(s) for s in sorted(init_sizes))))

to_write = pd.DataFrame(abundances).fillna(0).astype(int)
ofn = "%s/abundances.csv" % args.outdir
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
print('  writing to %s'%ofn)
to_write.to_csv(ofn)
