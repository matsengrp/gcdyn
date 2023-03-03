"""
Calculate the sequence abundance frequency for each input fasta and then write it as a CSV.
"""

import collections
import os
import sys
import pandas as pd
import argparse
import glob
import numpy

from Bio import SeqIO
from Bio.SeqUtils.CheckSum import seguid


ustr = """
python projects/gcdyn/scripts/abundance.py --indir /fh/fast/matsen_e/dralph/gcdyn/gcreplay-observed --outdir /fh/fast/matsen_e/dralph/gcdyn/tmp-out
"""
parser = argparse.ArgumentParser(usage=ustr)
parser.add_argument('--infiles', nargs='+')
parser.add_argument('--indir')
parser.add_argument('--outdir')
parser.add_argument('--mice', default=[1, 2, 3, 4, 5, 6], help='restrict to these mouse numbers')
parser.add_argument('--min-seqs', default=70, help='only keep GCs that have this many seqs')
parser.add_argument('--max-seqs', default=70, help='downsample to this many seqs any GCs that have more than this')
args = parser.parse_args()

if args.infiles is None:
    assert args.indir is not None
    args.infiles = glob.glob('%s/*.fasta'%args.indir)

# ----------------------------------------------------------------------------------------
def parse_name(fn):  # convert .fa name to mouse #, etc
    fn = os.path.basename(fn)
    # fn = 'annotated-PR-1-10-11-LA-122-GC.fasta'
    nlist = fn.split('-')
    if len(nlist) != 8:
        raise Exception('fn should have 7 - chars, but got %d: %s' % (fn.count('-'), fn))
    assert nlist[:2] == ['annotated', 'PR']
    flowcell = '-'.join(nlist[2:4])
    mouse = int(nlist[4])
    ln_loc = nlist[5]
    ln_id = nlist[6]
    assert nlist[7] == 'GC.fasta'
    return mouse, flowcell, ln_loc, ln_id

# ----------------------------------------------------------------------------------------
def bstr(mouse, flowcell, ln_loc, ln_id):
    return '-'.join([str(mouse), flowcell, ln_loc, ln_id])

# ----------------------------------------------------------------------------------------
skipped_mice, kept_mice, n_too_small, init_sizes = set(), set(), 0, []
abundances = {}

for fasta_path in args.infiles:
    mouse, flowcell, ln_loc, ln_id = parse_name(fasta_path)
    if args.mice is not None and mouse not in args.mice:
        skipped_mice.add(mouse)
        continue
    kept_mice.add(mouse)

    records = list(SeqIO.parse(fasta_path, "fasta"))
    if len(records) < args.min_seqs:
        n_too_small += 1
        continue
    if len(records) > args.max_seqs:
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

    # base, _ = os.path.splitext(fasta_path)
    base = bstr(mouse, flowcell, ln_loc, ln_id)
    abundances[base] = pd.Series(
        abundance_distribution.values(), index=abundance_distribution.keys()
    )

if len(skipped_mice) > 0:
    print '    skipped %d mice: %s' % (len(skipped_mice), ' '.join(str(s) for s in sorted(skipped_mice)))
if n_too_small > 0:
    print '    skipped %d files with fewer than %d seqs' % (n_too_small, args.min_seqs)
if len(init_sizes) > 0:
    print '    downsampled %d samples to %d from initial sizes: %s' % (len(init_sizes), args.max_seqs, ' '.join(str(s) for s in sorted(init_sizes)))
print '    kept %d samples from %d mice: %s' % (len(abundances), len(kept_mice), ' '.join(str(s) for s in sorted(kept_mice)))

to_write = pd.DataFrame(abundances).fillna(0).astype(int)
ofn = "%s/abundances.csv" % (args.outdir if args.outdir is not None else os.getcwd())
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
print('  writing to %s'%ofn)
# to_write.to_csv(ofn)
for base, vals in abundances.items():
    print base
    for x in vals:
        print x
    sys.exit()
