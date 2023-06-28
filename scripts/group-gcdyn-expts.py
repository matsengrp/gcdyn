#!/usr/bin/env python
import argparse
import os

# import colored_traceback.always
import numpy
import csv
import pandas as pd

from gcdyn import utils

# group deep learning predictions from --test-file (each of which correspond to the prediction on one tree) into "experiments" of
#  size --n-trees-per-expt, i.e. mimicking the case where we average over N trees in an experiment.
#  Note that this *ignores* values that are leftover after grouping into groups of --n-trees-per-expt

# ----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--test-file", required=True)
parser.add_argument("--outfile", required=True)
parser.add_argument("--n-trees-per-expt", required=True, type=int)
parser.add_argument("--params-to-predict", default=['xscale', 'xshift'], nargs='+')
args = parser.parse_args()

true_vals = {}
n_lines = 0
with open(args.test_file) as cfile:
    for cln in csv.DictReader(filter(lambda x: x[0] != "#", cfile)):
        n_lines += 1
        tvals, pvals = [tuple(float(cln['%s-%s'%(p, k)]) for p in args.params_to_predict) for k in ["truth", "predicted"]]
        if tvals not in true_vals:
            true_vals[tvals] = []
        true_vals[tvals].append(pvals)
print(
    "    read %d lines with %d different true values: %s"
    % (n_lines, len(true_vals), " ".join('('+', '.join("%.2f" % v for v in vals)+')' for vals in true_vals))
)

final_vals = []
for tvals, pvlist in true_vals.items():
    for ival, istart in enumerate(range(0, len(pvlist), args.n_trees_per_expt)):
        subvals = pvlist[istart : istart + args.n_trees_per_expt]
        fline = {"": ival}
        for ip, param in enumerate(args.params_to_predict):
            fline.update({"%s-truth"%param: tvals[ip], "%s-predicted"%param: numpy.median([vlist[ip] for vlist in subvals])})
        final_vals.append(fline)
print(
    "      grouped into %d final lines with value counts: %s"
    % (
        len(final_vals),
        "  ".join(
            "%s %d" % (v, len([x for x in final_vals if tuple(x["%s-truth"%p] for p in args.params_to_predict) == v]))
            for v in true_vals
        ),
    )
)

print("    writing %d lines to %s" % (len(final_vals), args.outfile))
if not os.path.exists(os.path.dirname(args.outfile)):
    os.makedirs(os.path.dirname(args.outfile))
with open(args.outfile, "w") as ofile:
    writer = csv.DictWriter(ofile, final_vals[0].keys())
    writer.writeheader()
    for fline in final_vals:
        writer.writerow(fline)

prdfs = {'test' : pd.read_csv(args.outfile)}
utils.make_dl_plots(prdfs, args.params_to_predict, os.path.dirname(args.outfile) + '/plots', xtra_txt=' (groups of %d trees)'%args.n_trees_per_expt)
