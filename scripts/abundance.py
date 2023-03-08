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

# ----------------------------------------------------------------------------------------
def hamming_distance(seq1, seq2):  # NOTE doesn't handle ambiguous bases
    assert len(seq1) == len(seq2)
    return sum(x != y for x, y in zip(seq1.upper(), seq2.upper()))

# ----------------------------------------------------------------------------------------
ustr = """
python abundance.py <input fasta files> [--outdir <outdir>]
"""
parser = argparse.ArgumentParser(usage=ustr)
parser.add_argument('infiles', nargs='+')
parser.add_argument('--outdir', default=os.getcwd())
parser.add_argument('--min-seqs', type=int, help='only keep GCs that have this many seqs')
parser.add_argument('--max-seqs', type=int, help='downsample to this many seqs any GCs that have more than this')
parser.add_argument('--naive-name', default='naive', help='name of naive sequence, so that it can be skipped')
parser.add_argument('--random-seed', type=int, default=1, help='random seed for subsampling')
args = parser.parse_args()

numpy.random.seed(args.random_seed)

n_too_small, init_sizes, n_removed = 0, [], 0
final_distrs = {k : {} for k in ['abundances', 'hdists']}  # hdist: hamming distance from naive (root-to-tip distance)

for fasta_path in args.infiles:
    records = list(SeqIO.parse(fasta_path, "fasta"))
    naive_record = None
    if args.naive_name is not None:
        n_before = len(records)
        new_records = []
        for rcd in records:
            if rcd.id == args.naive_name:  # doesn't check whether there's more than one naive sequence
                naive_record = rcd
            else:
                new_records.append(rcd)
        records = new_records
        if len(records) != n_before:
            n_removed += n_before - len(records)
    if args.min_seqs is not None and len(records) < args.min_seqs:
        n_too_small += 1
        continue
    if args.max_seqs is not None and len(records) > args.max_seqs:
        init_sizes.append(len(records))
        i_to_keep = numpy.random.choice(list(range(len(records))), size=args.max_seqs)  # i_to_keep is because numpy.random.choice barfs on <records> because each record handles the [] operator (@$$*$~$~!!)
        records = [records[i] for i in i_to_keep]

    # This dictionary will map sequence checksum to the list of squence ids that have that
    # sequence checksum.
    ids_by_checksum = {k : collections.defaultdict(list) for k in final_distrs}
    sequence_count = 0
    for record in records:
        sequence_count = sequence_count + 1
        for tkey in final_distrs:
            if tkey == 'abundances':
                vkey = seguid(record.seq)
            elif tkey == 'hdists':
                vkey = hamming_distance(record.seq, naive_record.seq)
            else:
                assert False
            ids_by_checksum[tkey][vkey].append(record.id)

    distr_dicts = {k : collections.defaultdict(int) for k in final_distrs}
    for tkey in final_distrs:
        for id_list in ids_by_checksum[tkey].values():
            id_count = len(id_list)
            distr_dicts[tkey][id_count] = distr_dicts[tkey][id_count] + 1
        assert sequence_count == sum(k * v for k, v in distr_dicts[tkey].items())

    base, _ = os.path.splitext(fasta_path)
    base = os.path.basename(base)
    for tkey in final_distrs:
        final_distrs[tkey][base] = pd.Series(
            distr_dicts[tkey].values(), index=distr_dicts[tkey].keys()
        )

if n_too_small > 0:
    print('    skipped %d files with fewer than %d seqs' % (n_too_small, args.min_seqs))
if len(init_sizes) > 0:
    print('    downsampled %d samples to %d from initial sizes: %s' % (len(init_sizes), args.max_seqs, ' '.join(str(s) for s in sorted(init_sizes))))
if n_removed > 0:
    print('    removed %d seqs with name \'%s\'' % (n_removed, args.naive_name))

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
for fstr in ['abundances', 'hdists']:
    to_write = pd.DataFrame(final_distrs[fstr]).fillna(0).astype(int)
    ofn = "%s/%s.csv" % (args.outdir, fstr)
    print('  writing to %s'%ofn)
    to_write.to_csv(ofn)
