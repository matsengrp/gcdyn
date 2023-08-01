#!/usr/bin/env python
import numpy as np
import argparse
import os
import sys
import operator
from sklearn import preprocessing
import sys

# import colored_traceback.always  # need to add this to installation stuff, i'm not sure how to do it atm
import time
import pickle
import pandas as pd
import random
import csv

from gcdyn import utils


# ----------------------------------------------------------------------------------------
def csvfn(smpl):
    return "%s/%s.csv" % (args.outdir, smpl)


# ----------------------------------------------------------------------------------------
def scale_vals(pvals, scaler=None, inverse=False, dont_scale=False, debug=True):
    """Scale pvals to mean 0 and variance 1. To reverse a scaling, pass in the original scaler and set inverse=True"""
    def get_lists(pvs):  # picks values from rows/columns to get a list of values for each parameter
        return [[plist[i] for plist in pvs] for i in range(len(args.params_to_predict))]
    def fnstr(pvs, fn):  # apply fn to each list from get_lists(), returns resulting combined str
        return ' '.join('%-7.3f'%fn(l) for l in get_lists(pvs))
    def lstr(lst):  # print nice str for values in list lst
        return ' '.join('%5.2f'%v for v in sorted(set(lst)))
    def print_debug(pvs, dstr):
        print('    %6s:  mean %s  var %s   min %s  max %s' % (dstr, fnstr(pvs, np.mean), fnstr(pvs, np.var), fnstr(pvs, min), fnstr(pvs, max)))  # , lstr(get_lists(pvs)[0])))
    if debug:
        print_debug(pvals, 'before')
    if scaler is None:
        scaler = preprocessing.StandardScaler().fit(pvals)
        # scaler = preprocessing.MinMaxScaler(feature_range=(-10, 10)).fit(pvals)
    if dont_scale:
        return pvals, scaler
    pscaled = scaler.inverse_transform(pvals) if inverse else scaler.transform(pvals)
    if debug:
        print_debug(pscaled, 'after')
    return pscaled, scaler


# ----------------------------------------------------------------------------------------
def get_prediction(smpl, model, smpldict, scaler):
    pred_resps = model.predict(smpldict[smpl]["trees"])
    true_resps = smpldict[smpl]["birth-responses"]
    dfdata = {
        "%s-%s" % (param, ptype): []
        for param in args.params_to_predict
        for ptype in ["truth", "predicted"]
    }
    assert len(pred_resps) == len(true_resps)
    pvals = [[float(resp.value) for resp in plist] for plist in pred_resps]
    pscaled, scaler = scale_vals(pvals, scaler=scaler, inverse=True, dont_scale=args.dont_scale_params)
    for tr_resp, prlist in zip(true_resps, pscaled):
        for ip, param in enumerate(args.params_to_predict):
            dfdata["%s-truth" % param].append(getattr(tr_resp, param))
            dfdata["%s-predicted" % param].append(prlist[ip])
    df = pd.DataFrame(dfdata)
    df.to_csv(csvfn(smpl))
    return df


# ----------------------------------------------------------------------------------------
def read_plot_csv():
    prdfs = {}
    for smpl in ["train", "test"]:
        prdfs[smpl] = pd.read_csv(csvfn(smpl))
    utils.make_dl_plots(prdfs, args.params_to_predict, args.outdir + "/plots")


# ----------------------------------------------------------------------------------------
def get_traintest_indices(samples):
    def is_avail(resp):
        for pname in args.test_param_vals:
            if resp._param_dict[pname] not in args.test_param_vals[pname]:
                return False  # if this response's e.g. xscale value isn't in the allowed values from the command line, skip it
        return True

    def print_stats():
        for pname in args.test_param_vals:
            all_vals = [r._param_dict[pname] for r in samples["birth-responses"]]
            val_counts = {v: all_vals.count(v) for v in set(all_vals)}
            n_remaining = (
                sum(val_counts.get(v, 0) for v in args.test_param_vals[pname]) - n_test
            )
            print(
                "      --test-%s-values: restricted to %d / %d %s values (%s), choosing %d / %d with these values from original value counts: %s"
                % (
                    pname,
                    len(args.test_param_vals[pname]),
                    len(set(all_vals)),
                    pname,
                    " ".join("%.2f" % v for v in args.test_param_vals[pname]),
                    n_test,
                    n_remaining + n_test,
                    "   ".join(
                        "%.2f %d" % (v, c)
                        for v, c in sorted(
                            val_counts.items(), key=operator.itemgetter(0)
                        )
                    ),
                )
            )

    n_trees = len(samples["trees"])
    n_test = round((1.0 - args.train_frac) * n_trees)
    idxs = {}
    if args.test_param_vals is None:
        idxs["train"] = random.sample(range(n_trees), round(args.train_frac * n_trees))
        idxs["test"] = [i for i in range(n_trees) if i not in idxs["train"]]
    else:
        avail_indices = [
            i for i, r in enumerate(samples["birth-responses"]) if is_avail(r)
        ]  # indices of all trees with the specified xscale value
        idxs["test"] = random.sample(
            avail_indices, n_test
        )  # note that this'll change the distribution of xscale values in the training sample (so make sure that n_test isn't a large fraction of the trees with each xscale value
        idxs["train"] = [i for i in range(n_trees) if i not in idxs["test"]]
        print_stats()
    print("    chose %d test samples (from %d total)" % (n_test, n_trees))
    return idxs


# ----------------------------------------------------------------------------------------
def train_and_test():
    from gcdyn.models import NeuralNetworkModel
    from gcdyn.poisson import ConstantResponse
    from gcdyn import encode

    rfn, tfn, sfn = ['%s/%s' % (args.indir, s) for s in ['responses.pkl', 'encoded-trees.npy', 'summary-stats.csv']]
    with open(rfn, "rb") as rfile:
        pklfo = pickle.load(rfile)
    samples = {k + "-responses": [tfo[k] for tfo in pklfo] for k in ["birth", "death"]}
    samples["trees"] = encode.read_trees(tfn)
    sstats = []
    with open(sfn) as sfile:
        reader = csv.DictReader(sfile)
        for line in reader:
            sstats.append(line)
# TODO not using sstats yet, need to implement a way to scale any future summary stats that depend on branch length
    print(
        "    read %d trees from %s (%d responses from %s)"
        % (len(samples["trees"]), tfn, len(pklfo), rfn)
    )
    print(
        "      first response pair:\n        birth: %s\n        death: %s"
        % (samples["birth-responses"][0], samples["death-responses"][0])
    )

    idxs = get_traintest_indices(samples)
    smpldict = {}
    for smpl in ["train", "test"]:
        smpldict[smpl] = {
            key: [val[i] for i in idxs[smpl]] for key, val in samples.items()
        }
    print(
        "      N trees: %d train   %d test"
        % (len(smpldict["train"]["trees"]), len(smpldict["test"]["trees"]))
    )

    pscaled, scalers = {}, {}  # scaled parameters and scalers
    for smpl in smpldict:
        pvals = [[getattr(birth_resp, pname) for pname in args.params_to_predict] for birth_resp in smpldict[smpl]["birth-responses"]]
        pscaled[smpl], scalers[smpl] = scale_vals(pvals, dont_scale=args.dont_scale_params)
    if args.use_trivial_encoding:
        for smpl in smpldict:
            encode.trivialize_encodings(smpldict[smpl]['trees'], pscaled[smpl], noise=True) #, n_debug=3)

    model = NeuralNetworkModel(
        smpldict["train"]["trees"], [[ConstantResponse(v) for v in vlist] for vlist in pscaled['train']], network_layers=args.model_size
    )
    model.fit(epochs=args.epochs)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    print("  writing train/test results to %s" % args.outdir)

    prdfs = {}
    for smpl in ["train", "test"]:
        prdfs[smpl] = get_prediction(smpl, model, smpldict, scalers['train'])
    utils.make_dl_plots(prdfs, args.params_to_predict, args.outdir + "/plots")

    print("    total dl inference time: %.1f sec" % (time.time() - start))


# ----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--indir",
    required=True,
    help="input directory with simulation output (uses encoded trees .npy, summary stats .csv, and response .pkl files)",
)
parser.add_argument("--outdir", required=True, help="output directory")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument(
    "--train-frac", type=float, default=0.8, help="train on this fraction of the trees"
)
parser.add_argument(
    "--test-xscale-values",
    type=float,
    nargs="+",
    help="if set, choose test samples only from among those with this (birth) xscale value.",
)
parser.add_argument(
    "--test-xshift-values",
    type=float,
    nargs="+",
    help="if set, choose test samples only from among those with this (birth) xshift value.",
)
parser.add_argument(
    "--model-size",
    default="tiny",
    choices=["small", "tiny", 'trivial', None],
    help="Parameters from the birth model that we should try to predict.",
)
parser.add_argument(
    "--params-to-predict",
    default=["xscale", "xshift"],
    nargs="+",
    choices=["xscale", "xshift"],
)
parser.add_argument(
    "--test",
    action="store_true",
    help="sets things to be super fast, so not useful for real inference, but just to check if things are running properly",
)
parser.add_argument("--random-seed", default=0, type=int, help="random seed")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--use-trivial-encoding", action="store_true")
parser.add_argument("--dont-scale-params", action="store_true")

start = time.time()
args = parser.parse_args()
args.test_param_vals = None
if args.test_xscale_values is not None or args.test_xshift_values is not None:
    args.test_param_vals = {}
    if args.test_xscale_values is not None:
        args.test_param_vals.update({"xscale": args.test_xscale_values})
        delattr(args, "test_xscale_values")
    if args.test_xshift_values is not None:
        args.test_param_vals.update({"xshift": args.test_xshift_values})
        delattr(args, "test_xshift_values")
if args.test:
    args.epochs = 10

random.seed(args.random_seed)
np.random.seed(args.random_seed)


# ----------------------------------------------------------------------------------------
if os.path.exists(csvfn("test")) and not args.overwrite:
    print(
        "    csv files already exist, so just replotting (override with --overwrite): %s"
        % csvfn("test")
    )
    read_plot_csv()
    sys.exit(0)

train_and_test()
