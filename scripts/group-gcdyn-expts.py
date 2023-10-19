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
parser.add_argument("--indir", required=True)
parser.add_argument("--outdir", required=True)
parser.add_argument("--n-trees-per-expt", required=True, type=int)
parser.add_argument("--params-to-predict", default=["xscale", "xshift"], nargs="+")
args = parser.parse_args()


# ----------------------------------------------------------------------------------------
def read_values(vtype):
    vtvals = {}
    n_lines = 0
    with open("%s/%s.csv" % (args.indir, vtype)) as cfile:
        for cln in csv.DictReader(filter(lambda x: x[0] != "#", cfile)):
            n_lines += 1
            tvals, pvals = [
                tuple(float(cln["%s-%s" % (p, k)]) for p in args.params_to_predict)
                for k in ["truth", "predicted"]
            ]
            if tvals not in vtvals:
                vtvals[tvals] = []
            vtvals[tvals].append(pvals)
    print(
        "    %s: read %d lines with %d different true values: %s"
        % (
            vtype,
            n_lines,
            len(vtvals),
            " ".join(
                "(" + " ".join("%.2f," % v for v in vals) + ")"
                for vals in sorted(vtvals)
            ),
        )
    )
    return vtvals


# ----------------------------------------------------------------------------------------
vtypes = ["train", "test"]

prdfs = {}
for vtp in vtypes:
    true_vals = read_values(vtp)

    final_vals = []
    for tvals, pvlist in true_vals.items():
        for ival, istart in enumerate(range(0, len(pvlist), args.n_trees_per_expt)):
            subvals = pvlist[istart : istart + args.n_trees_per_expt]
            fline = {"": ival}
            for ip, param in enumerate(args.params_to_predict):
                fline.update(
                    {
                        "%s-truth" % param: tvals[ip],
                        "%s-predicted"
                        % param: numpy.median([vlist[ip] for vlist in subvals]),
                    }
                )
            final_vals.append(fline)

    def tstr(tpl):
        return "(" + " ".join(["%.2f," % v for v in tpl]) + ")"

    print(
        "       grouped into %d final lines with value counts: %s"
        % (
            len(final_vals),
            "  ".join(
                "%s %d"
                % (
                    tstr(v),
                    len(
                        [
                            x
                            for x in final_vals
                            if tuple(x["%s-truth" % p] for p in args.params_to_predict)
                            == v
                        ]
                    ),
                )
                for v in sorted(true_vals)
            ),
        )
    )

    ofn = "%s/%s.csv" % (args.outdir, vtp)
    print("       writing %d lines to %s" % (len(final_vals), ofn))
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    with open(ofn, "w") as ofile:
        writer = csv.DictWriter(ofile, final_vals[0].keys())
        writer.writeheader()
        for fline in final_vals:
            writer.writerow(fline)

    prdfs[vtp] = pd.read_csv(ofn)

utils.make_dl_plots(
    prdfs,
    args.params_to_predict,
    args.outdir + "/plots",
    xtra_txt=" (groups of %d trees)" % args.n_trees_per_expt,
)
