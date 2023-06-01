"""
Calculate the sequence abundance frequency for each input fasta and then write it as a CSV.
"""

import collections
import os
import pandas as pd
import argparse
import numpy
import csv
import itertools

from Bio import SeqIO
from Bio.SeqUtils.CheckSum import seguid


# ----------------------------------------------------------------------------------------
def hamming_distance(seq1, seq2):  # NOTE doesn't handle ambiguous bases
    assert len(seq1) == len(seq2)
    return sum(x != y for x, y in zip(seq1.upper(), seq2.upper()))


# ----------------------------------------------------------------------------------------
ustr = """
Collect a variety of abundance and mutation information about the sequences in a list of fasta files.
Writes three files:
        abundances.csv: tabular abundance info, one column for each input fasta
        hdists.csv: hamming distance to naive sequence (i.e. N mutations), colon-separated list over sequences, one row for each input file
        max-abdn-shm.csv: N mutations in sequences with maximum abundance (median if ties), one row for each input file

usage: python abundance.py <input fasta files> [--outdir <outdir>]

"""
parser = argparse.ArgumentParser(usage=ustr)
parser.add_argument("infiles", nargs="+")
parser.add_argument("--outdir", default=os.getcwd())
parser.add_argument(
    "--min-seqs", type=int, help="only keep GCs that have this many seqs"
)
parser.add_argument(
    "--max-seqs",
    type=int,
    help="downsample to this many seqs any GCs that have more than this",
)
parser.add_argument(
    "--naive-name",
    default="naive",
    help="name of naive sequence, so that it can be skipped",
)
parser.add_argument(
    "--random-seed", type=int, default=1, help="random seed for subsampling"
)
args = parser.parse_args()

numpy.random.seed(args.random_seed)

n_too_small, init_sizes, n_removed = 0, [], 0
abundances = {}
fdicts = {"hdists": {}, "max-abdn-shm": {}}

for fasta_path in args.infiles:
    records = list(SeqIO.parse(fasta_path, "fasta"))
    naive_record = None
    if args.naive_name is not None:
        n_before = len(records)
        new_records = []
        for rcd in records:
            if (
                rcd.id == args.naive_name
            ):  # doesn't check whether there's more than one naive sequence
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
        i_to_keep = numpy.random.choice(
            list(range(len(records))), size=args.max_seqs
        )  # i_to_keep is because numpy.random.choice barfs on <records> because each record handles the [] operator (@$$*$~$~!!)
        records = [records[i] for i in i_to_keep]

    # This dictionary will map sequence checksum to the list of squence ids that have that
    # sequence checksum.
    ids_by_checksum = collections.defaultdict(list)
    hdvals, hd_dict = (
        [],
        {},
    )  # list of hamming distance to naive for each sequence (hd_dict is just for max_abdn below)
    sequence_count = 0
    for record in records:
        sequence_count = sequence_count + 1
        ids_by_checksum[seguid(record.seq)].append(record.id)
        hdvals.append(hamming_distance(record.seq, naive_record.seq))
        hd_dict[record.id] = hdvals[-1]

    abundance_distribution = collections.defaultdict(int)
    for id_list in ids_by_checksum.values():
        id_count = len(id_list)
        abundance_distribution[id_count] = abundance_distribution[id_count] + 1
    assert sequence_count == sum(k * v for k, v in abundance_distribution.items())
    for amax, max_abdn_idlists in itertools.groupby(
        sorted(ids_by_checksum.values(), key=len, reverse=True), key=len
    ):
        # print(amax, [len(l) for l in max_abdn_idlists])
        break  # just want the first one (max abundance)

    base, _ = os.path.splitext(fasta_path)
    base = os.path.basename(base)
    abundances[base] = pd.Series(
        abundance_distribution.values(), index=abundance_distribution.keys()
    )

    fdicts["hdists"][base] = hdvals
    fdicts["max-abdn-shm"][base] = [
        int(numpy.median([hd_dict[u] for x in max_abdn_idlists for u in x]))
    ]

if n_too_small > 0:
    print("    skipped %d files with fewer than %d seqs" % (n_too_small, args.min_seqs))
if len(init_sizes) > 0:
    print(
        "    downsampled %d samples to %d from initial sizes: %s"
        % (len(init_sizes), args.max_seqs, " ".join(str(s) for s in sorted(init_sizes)))
    )
if n_removed > 0:
    print("    removed %d seqs with name '%s'" % (n_removed, args.naive_name))

print("  writing to %s" % args.outdir)
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

to_write = pd.DataFrame(abundances).fillna(0).astype(int)
to_write.to_csv("%s/abundances.csv" % args.outdir)

for fstr, fdct in fdicts.items():
    with open("%s/%s.csv" % (args.outdir, fstr), "w") as cfile:
        writer = csv.DictWriter(cfile, ["fbase", "vlist"])
        writer.writeheader()
        for fbase, vlist in fdicts[fstr].items():
            writer.writerow({"fbase": fbase, "vlist": ":".join(str(v) for v in vlist)})
