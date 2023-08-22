#!/usr/bin/env python
import numpy as np
import argparse
import os
import sys
import operator
from sklearn import preprocessing
import sys

import colored_traceback.always  # need to add this to installation stuff, i'm not sure how to do it atm
import time
import pickle
import pandas as pd
import random
import csv
import copy

from gcdyn import utils, encode

# ----------------------------------------------------------------------------------------
sum_stat_scaled = {'total_branch_length' : True}  # whether to scale summary stats with branch length
smplist = ["train", "test"]

# ----------------------------------------------------------------------------------------
def csvfn(smpl):
    return "%s/%s.csv" % (args.outdir, smpl)


# ----------------------------------------------------------------------------------------
def scale_vals(smpl, pvals, scaler=None, inverse=False, debug=True):
    """Scale pvals for a single sample to mean 0 and variance 1. To reverse a scaling, pass in the original scaler and set inverse=True"""
    # ----------------------------------------------------------------------------------------
    def get_lists(pvs):  # picks values from rows/columns to get a list of values for each parameter
        return [[plist[i] for plist in pvs] for i in range(len(args.params_to_predict))]
    def fnstr(pvs, fn):  # apply fn to each list from get_lists(), returns resulting combined str
        return ' '.join('%7.3f'%fn(l) for l in get_lists(pvs))
    def lstr(lst):  # print nice str for values in list lst
        return ' '.join('%5.2f'%v for v in sorted(set(lst)))
    def print_debug(pvs, dstr):
        print('%s %s%s %s%s' % (('        %7s'%smpl) if dstr=='before' else '   ', fnstr(pvs, np.mean), fnstr(pvs, np.var), fnstr(pvs, min), fnstr(pvs, max)), end='' if dstr == 'before' else '\n')

    # ----------------------------------------------------------------------------------------
    if debug and smpl == smplist[0]:
        print('                          before                             after')
        print('                   mean   var     min   max         mean   var     min   max')
    if debug:
        print_debug(pvals, 'before')
    if scaler is None:
        scaler = preprocessing.StandardScaler().fit(pvals)
        # scaler = preprocessing.MinMaxScaler(feature_range=(0, 10)).fit(pvals)
    if args.dont_scale_params:
        return copy.copy(pvals), scaler
    pscaled = scaler.inverse_transform(pvals) if inverse else scaler.transform(pvals)
    if debug:
        print_debug(pscaled, 'after')
    return pscaled, scaler


# ----------------------------------------------------------------------------------------
def get_prediction(smpl, model, smpldict, scaler):
    pred_resps = model.predict(smpldict[smpl]["trees"])
    true_resps, true_sstats = [smpldict[smpl][tk] for tk in ["birth-responses", 'sstats']]
    dfdata = {
        "%s-%s" % (param, ptype): []
        for param in args.params_to_predict
        for ptype in ["truth", "predicted"]
    }
    assert len(pred_resps) == len(true_resps)
    pvals = [[float(resp.value) for resp in plist] for plist in pred_resps]
    pscaled, _ = scale_vals(smpl, pvals, scaler=scaler, inverse=True)
    for tr_resp, prlist, sum_stats in zip(true_resps, pscaled, true_sstats):
        for ip, param in enumerate(args.params_to_predict):
            dfdata["%s-truth" % param].append(get_param(param, tr_resp, sum_stats))
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
    n_trees = len(samples["trees"])
    n_test = round((1.0 - args.train_frac) * n_trees)
    idxs = {}
    idxs["train"] = range(round(args.train_frac * n_trees))
    print('    taking first %d trees to train' % len(idxs['train']))
    idxs["test"] = [i for i in range(n_trees) if i not in idxs["train"]]
    print("    chose %d test samples (from %d total)" % (n_test, n_trees))
    return idxs


# ----------------------------------------------------------------------------------------
def get_param(pname, bresp, sts):
    if pname == 'total_branch_length':
        return float(sts['total_branch_length'])
    elif hasattr(bresp, pname):
        return getattr(bresp, pname)
    else:
        assert False

# ----------------------------------------------------------------------------------------
def write_traintest_samples(smpldict):
    for smpl in smplist:
        subdict = smpldict[smpl]
        responses = [{k : subdict[k + '-responses'][i] for k in ['birth', 'death']} for i in range(len(subdict['trees']))]
        encode.write_training_files('%s/%s-samples' % (args.outdir, smpl), subdict['trees'], subdict['sstats'], responses, dbgstr=smpl)

# ----------------------------------------------------------------------------------------
def train_and_test():
    from gcdyn.models import NeuralNetworkModel
    from gcdyn.poisson import ConstantResponse

    # read from various input files
    rfn, tfn, sfn = ['%s/%s' % (args.indir, s) for s in ['responses.pkl', 'encoded-trees.npy', 'summary-stats.csv']]
    with open(rfn, "rb") as rfile:
        pklfo = pickle.load(rfile)
    samples = {k + "-responses": [tfo[k] for tfo in pklfo] for k in ["birth", "death"]}
    samples["trees"] = encode.read_trees(tfn)
    samples['sstats'] = []
    with open(sfn) as sfile:
        reader = csv.DictReader(sfile)
        for line in reader:
            samples['sstats'].append(line)
    print(
        "    read %d trees from %s (%d responses from %s)"
        % (len(samples["trees"]), tfn, len(pklfo), rfn)
    )
    print(
        "      first response pair:\n        birth: %s\n        death: %s"
        % (samples["birth-responses"][0], samples["death-responses"][0])
    )

    # separate train/test samples
    idxs = get_traintest_indices(samples)
    smpldict = {}  # separate train/test trees and responses by index
    for smpl in smplist:
        smpldict[smpl] = {
            key: [val[i] for i in idxs[smpl]] for key, val in samples.items()
        }
    print(
        "      N trees: %s"
        % "   ".join("%s %d" % (s, len(smpldict[s]["trees"])) for s in smplist)
    )

    write_traintest_samples(smpldict)

    # handle various scaling/re-encoding stuff
    pscaled, scalers = {}, {}  # scaled parameters and scalers
    for smpl in smplist:
        pvals = [[get_param(pname, bresp, sts) for pname in args.params_to_predict] for bresp, sts in zip(smpldict[smpl]['birth-responses'], smpldict[smpl]['sstats'])]
        pscaled[smpl], scalers[smpl] = scale_vals(smpl, pvals)
    if args.use_trivial_encoding:  # silly encodings for testing that essentially train on the output values
        for smpl in smplist:
            encode.trivialize_encodings(smpldict[smpl]['trees'], pscaled[smpl], noise=True) #, n_debug=3)

    # train
    model = NeuralNetworkModel(
        smpldict["train"]["trees"],
        [[ConstantResponse(v) for v in vlist] for vlist in pscaled['train']],
        bundle_size=args.bundle_size,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        ema_momentum=args.ema_momentum
    )
    model.fit(epochs=args.epochs, validation_split=0.1)

    # evaluate/predict
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
parser.add_argument("--indir", required=True, help="input directory with simulation output (uses encoded trees .npy, summary stats .csv, and response .pkl files)")
parser.add_argument("--outdir", required=True, help="output directory")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--bundle-size", type=int, default=50)
parser.add_argument("--dropout-rate", type=float, default=0.2)
parser.add_argument("--learning-rate", type=float, default=0.01)
parser.add_argument("--ema-momentum", type=float, default=0.9)
parser.add_argument("--train-frac", type=float, default=0.8, help="train on this fraction of the trees")
parser.add_argument("--params-to-predict", default=["xscale", "xshift"], nargs="+", choices=["xscale", "xshift"] + [k for k in sum_stat_scaled])
parser.add_argument("--test", action="store_true", help="sets things to be super fast, so not useful for real inference, but just to check if things are running properly")
parser.add_argument("--random-seed", default=0, type=int, help="random seed")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--use-trivial-encoding", action="store_true")
parser.add_argument("--dont-scale-params", action="store_true")

start = time.time()
args = parser.parse_args()
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
