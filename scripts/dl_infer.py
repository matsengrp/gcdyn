#!/usr/bin/env python
import numpy as np
import argparse
import os
import sys
from sklearn import preprocessing

import colored_traceback.always  # noqa: F401
import time
import pickle
import pandas as pd
import random
import csv
import copy
import collections

from gcdyn import utils, encode
from gcdyn.nn import NeuralNetworkModel
from gcdyn.poisson import ConstantResponse

# ----------------------------------------------------------------------------------------
sum_stat_scaled = {
    "total_branch_length": True
}  # whether to scale summary stats with branch length
smplist = ["train", "test"]


# ----------------------------------------------------------------------------------------
def csvfn(args, smpl):
    return "%s/%s.csv" % (args.outdir, smpl)


# ----------------------------------------------------------------------------------------
# fmt: off
def scale_vals(args, in_pvals, scaler=None, inverse=False, smpl='', debug=True):
    """Scale in_pvals for a single sample to mean 0 and variance 1. To reverse a scaling, pass in the original scaler and set inverse=True"""
    # ----------------------------------------------------------------------------------------
    def print_debug(pvals_before, pvals_scaled):
        def get_lists(pvs,):  # picks values from rows/columns to get a list of values for each parameter
            return [[plist[ivar] for plist in pvs]]
        def fnstr(pvs, fn):  # apply fn to each list from get_lists(), returns resulting combined str
            return " ".join("%7.2f" % fn(vl) for vl in get_lists(pvs))
        for ivar, vname in enumerate(args.params_to_predict):
            for dstr, pvs in zip(("before", "after"), (pvals_before, pvals_scaled)):
                bstr = "   " if dstr != "before" else "      %10s %7s" % (vname, smpl)
                print("%s %s%s %s%s" % (bstr, fnstr(pvs, np.mean), fnstr(pvs, np.var), fnstr(pvs, min), fnstr(pvs, max), ), end="" if dstr == "before" else "\n")
    # ----------------------------------------------------------------------------------------
    if debug:  # and smpl == smplist[0]:
        print("    %sscaling %d variables: %s" % ("reverse " if inverse else "", len(args.params_to_predict), args.params_to_predict,))
        print("                                  before                             after")
        print("                           mean   var     min   max         mean   var     min   max")
    if scaler is None:
        scaler = preprocessing.StandardScaler().fit(in_pvals)
        # scaler = preprocessing.MinMaxScaler(feature_range=(0, 10)).fit(in_pvals)
    if args.dont_scale_params:
        return copy.copy(in_pvals), scaler
    sc_pvals = scaler.inverse_transform(in_pvals) if inverse else scaler.transform(in_pvals)
    if debug:
        print_debug(in_pvals, sc_pvals)
    return sc_pvals, scaler


# ----------------------------------------------------------------------------------------
def collapse_bundles(args, resps, sstats):
    resps = [
        NeuralNetworkModel._collapse_identical_list(resps[i : i + args.bundle_size])
        for i in range(0, len(resps), args.bundle_size)
    ]
    sstats = [
        {
            tkey: (min if tkey == "tree" else np.mean)(
                [float(sstats[i + j][tkey]) for j in range(args.bundle_size)]
            )
            for tkey in sstats[i]
        }
        for i in range(0, len(sstats), args.bundle_size)
    ]  # mean of each summary stat over trees in each bundle ('tree' key is an index, so take min/index of first one)
    return resps, sstats

# ----------------------------------------------------------------------------------------
def write_prediction(args, pred_resps, true_resps=None, true_sstats=None, scaler=None, smpl=None):
    pvals = [[float(resp.value) for resp in plist] for plist in pred_resps]
    punscaled, _ = scale_vals(args, pvals, smpl=smpl, scaler=scaler, inverse=True)
    dfdata = {  # make empty df
        "%s-%s" % (param, ptype): []
        for param in args.params_to_predict
        for ptype in (["predicted"] if true_resps is None else ["truth", "predicted"])
    }
    assert true_resps is None or len(punscaled) == len(true_resps)
    for itr, prlist in enumerate(punscaled):
        for ip, param in enumerate(args.params_to_predict):
            dfdata["%s-predicted" % param].append(prlist[ip])
            if true_resps is not None:
                dfdata["%s-truth" % param].append(get_pval(param, true_resps[itr], true_sstats[itr]))
    df = pd.DataFrame(dfdata)
    print("  writing %s results to %s" % (smpl, args.outdir))
    df.to_csv(csvfn(args, smpl))
    return df

# ----------------------------------------------------------------------------------------
def get_prediction(args, model, spld, scaler, smpl=None):
    pred_resps = model.predict(spld["trees"])  # note that this returns constant response fcns that are just holders for the predicted values (i.e. don't directly relate to true/input response fcns)
    true_resps, true_sstats = None, None
    if args.is_simu:
        true_resps, true_sstats = [spld[tk] for tk in ["birth-responses", "sstats"]]
        if args.bundle_size > 1:
            true_resps, true_sstats = collapse_bundles(args, true_resps, true_sstats)
        assert len(pred_resps) == len(true_resps)
    df = write_prediction(args, pred_resps, true_resps=true_resps, true_sstats=true_sstats, smpl=smpl, scaler=scaler)
    return df

# ----------------------------------------------------------------------------------------
def read_plot_csv(args):
    prdfs = {}
    for smpl in ["train", "test"]:
        prdfs[smpl] = pd.read_csv(csvfn(args, smpl))
    utils.make_dl_plots(
        prdfs,
        args.params_to_predict,
        args.outdir + "/plots",
        validation_split=args.validation_split,
    )


# ----------------------------------------------------------------------------------------
def get_traintest_indices(args, samples):
    n_trees = len(samples["trees"])
    n_test = round((1.0 - args.train_frac) * n_trees)
    idxs = {}
    idxs["train"] = range(round(args.train_frac * n_trees))
    print("    taking first %d trees to train" % len(idxs["train"]))
    idxs["test"] = [i for i in range(n_trees) if i not in idxs["train"]]
    print("    chose %d test samples (from %d total)" % (n_test, n_trees))
    return idxs


# ----------------------------------------------------------------------------------------
def get_pval(pname, bresp, sts):  # get parameter value from response fcn
    if pname == "total_branch_length":
        return float(sts["total_branch_length"])
    elif hasattr(bresp, pname):
        return getattr(bresp, pname)
    else:
        assert False


# ----------------------------------------------------------------------------------------
def get_pvlists(args, samples):  # rearrange + convert response fcn lists to lists of parameter values
    pvals = [
        [get_pval(pname, bresp, sts) for pname in args.params_to_predict]
        for bresp, sts in zip(
            samples["birth-responses"], samples["sstats"]
        )
    ]
    return pvals


# ----------------------------------------------------------------------------------------
def write_traintest_samples(args, smpldict):
    for smpl in smplist:
        subdict = smpldict[smpl]
        responses = [
            {k: subdict[k + "-responses"][i] for k in ["birth", "death"]}
            for i in range(len(subdict["trees"]))
        ]
        encode.write_training_files(
            "%s/%s-samples" % (args.outdir, smpl),
            subdict["trees"],
            subdict["sstats"],
            responses,
            dbgstr=smpl,
        )


# ----------------------------------------------------------------------------------------
def read_tree_files(args):
    # ----------------------------------------------------------------------------------------
    def check_bundles(samples, debug=False):
        rnames = ['birth', 'death']
        ibund, broken_bundles = -1, []
        bundle_rates, tree_indices = {n : [] for n in rnames}, []
        for iresp, (br, dr) in enumerate(zip(samples["birth-responses"], samples["death-responses"])):
            if iresp % args.bundle_size == 0:
                for rn in rnames:
                    if len(bundle_rates[rn]) > 1:
                        broken_bundles.append(ibund)
                        print('    %s bundle with index %d (tree indices %d to %d) has multiple %s rates:\n      %s' % (utils.color("yellow", "warning"), ibund, min(tree_indices), max(tree_indices), rn, '\n      '.join(str(r) for r in bundle_rates[rn])))
                        # print(' '.join(str(i) for i in tree_indices))  # maybe want to print all broken tree indices for deletion by hand
                bundle_rates['birth'], bundle_rates['death'] = [br], [dr]
                tree_indices = []
                ibund += 1
                if debug:
                    print('  bundle %d (size %d)' % (ibund, args.bundle_size))
            tree_indices.append(iresp)
            if debug:
                print('      %7d  %s  %s' % (iresp, br, dr))
            for this_rate, rn in zip([br, dr], rnames):
                if this_rate != bundle_rates[rn][0]:
                    if br not in bundle_rates[rn]:
                        bundle_rates[rn].append(br)
                    if debug:
                        print('        %s birth rate not equal to last one' % (utils.color("yellow", "warning")))
        if len(broken_bundles) > 0:
            raise Exception('%d broken bundles, see above' % len(broken_bundles))
    # ----------------------------------------------------------------------------------------
    # read from various input files
    rfn, tfn, sfn = [
        "%s/%s" % (args.indir, s)
        for s in ["responses.pkl", "encoded-trees.npy", "summary-stats.csv"]
    ]
    with open(rfn, "rb") as rfile:
        pklfo = pickle.load(rfile)
    samples = {k + "-responses": [tfo[k] for tfo in pklfo] for k in ["birth", "death"]}
    samples["trees"] = encode.read_trees(tfn)
    samples["sstats"] = []
    with open(sfn) as sfile:
        reader = csv.DictReader(sfile)
        for line in reader:
            samples["sstats"].append(line)
    print("    read %d trees from %s (%d responses from %s)" % (len(samples["trees"]), tfn, len(pklfo), rfn))
    print("      first response pair:\n        birth: %s\n        death: %s" % (samples["birth-responses"][0], samples["death-responses"][0]))
    check_bundles(samples)
    return samples


# ----------------------------------------------------------------------------------------
def train_and_test(args, start_time):
    samples = read_tree_files(args)

    # separate train/test samples
    idxs = get_traintest_indices(args, samples)
    smpldict = {}  # separate train/test trees and responses by index
    for smpl in smplist:
        smpldict[smpl] = {
            key: [val[i] for i in idxs[smpl]] for key, val in samples.items()
        }
    print("      N trees: %s" % "   ".join("%s %d" % (s, len(smpldict[s]["trees"])) for s in smplist))

    write_traintest_samples(args, smpldict)

    # handle various scaling/re-encoding stuff
    pscaled, scalers = {}, {}  # scaled parameters and scalers
    for smpl in smplist:
        pvals = get_pvlists(args, smpldict[smpl])
        pscaled[smpl], scalers[smpl] = scale_vals(args, pvals, smpl=smpl)
    if args.use_trivial_encoding:  # silly encodings for testing that essentially train on the output values
        for smpl in smplist:
            encode.trivialize_encodings(smpldict[smpl]["trees"], pscaled[smpl], noise=True)  # , n_debug=3)

    # train
    responses = [[ConstantResponse(v) for v in vlist] for vlist in pscaled["train"]]
    model = NeuralNetworkModel(responses[0], bundle_size=args.bundle_size)
    model.build_model(
        smpldict["train"]["trees"],
        responses,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        ema_momentum=args.ema_momentum,
        prebundle_layer_cfg=args.prebundle_layer_cfg,
    )
    model.fit(epochs=args.epochs, validation_split=args.validation_split)
    model.network.save('%s/model.h5' % (args.outdir))

    # evaluate/predict
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    prdfs = {}
    for smpl in ["train", "test"]:
        prdfs[smpl] = get_prediction(args, model, smpldict[smpl], scalers["train"], smpl=smpl)
    utils.make_dl_plots(
        prdfs,
        args.params_to_predict,
        args.outdir + "/plots",
        validation_split=args.validation_split,
    )

    print("    total dl inference time: %.1f sec" % (time.time() - start_time))


# ----------------------------------------------------------------------------------------
def infer(args, start_time):
    samples = read_tree_files(args)
    pvals = get_pvlists(args, samples)
    pscaled, scaler = scale_vals(args, pvals)
    example_response_list = [ConstantResponse(v) for v in pscaled[0]]
    model = NeuralNetworkModel(example_response_list, bundle_size=args.bundle_size)
    model.load(args.model_file)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    prdf = get_prediction(args, model, samples, scaler, smpl='infer')
    utils.make_dl_plots(
        {'infer' : prdf},
        args.params_to_predict,
        args.outdir + "/plots",
        validation_split=0,
    )

    print("    total dl inference time: %.1f sec" % (time.time() - start_time))

# ----------------------------------------------------------------------------------------
def get_parser():
    helpstr = """
    Infer affinity response function on gcdyn simulation using deep learning neural networks.
    Two actions: <train> trains and tests a new dl model on the sample in the input dir, whereas
    <infer> infers parameters with an existing dl model.
    Example usage:
        gcd-dl train --indir <input dir> --outdir <output dir>
        gcd-dl infer --indir <input dir> --outdir <output dir> --model-file <model file>
    """

    class MultiplyInheritedFormatter(
        argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MultiplyInheritedFormatter, description=helpstr
    )
    parser.add_argument("action", choices=['train', 'infer'])
    parser.add_argument("--indir", required=True, help="training input directory with gcdyn simulation output (uses encoded trees .npy, summary stats .csv, and response .pkl files)", )
    parser.add_argument("--model-file", help="file with saved deep learning model for inference")
    parser.add_argument("--outdir", required=True, help="output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bundle-size", type=int, default=50)
    parser.add_argument("--dropout-rate", type=float, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--prebundle-layer-cfg", default='default') #, choices=['default', 'small', 'big', 'huge'])
    parser.add_argument("--train-frac", type=float, default=0.8, help="train on this fraction of the trees")
    parser.add_argument("--validation-split", type=float, default=0.1, help="fraction of training sample to tell keras to hold out for validation during training")
    parser.add_argument("--params-to-predict", default=["xscale", "xshift", "yscale"], nargs="+", choices=["xscale", "xshift", "yscale"] + [k for k in sum_stat_scaled])
    parser.add_argument("--test", action="store_true", help="sets things to be super fast, so not useful for real inference, but just to check if things are running properly")
    parser.add_argument("--is-simu", action="store_true", help="set to this if running on simulation")
    parser.add_argument("--random-seed", default=0, type=int, help="random seed")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-trivial-encoding", action="store_true")
    parser.add_argument("--dont-scale-params", action="store_true")
    return parser


# ----------------------------------------------------------------------------------------
def main():
    parser = get_parser()
    start_time = time.time()
    args = parser.parse_args()
    if args.test:
        args.epochs = 10

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    import tensorflow as tf  # this is super slow, don't want to wait for this to get help message

    tf.keras.utils.set_random_seed(args.random_seed)

    if args.action == 'train':
        if os.path.exists(csvfn(args, "test")) and not args.overwrite:
            print("    csv files already exist, so just replotting (override with --overwrite): %s" % csvfn(args, "test"))
            read_plot_csv(args)
            sys.exit(0)
        train_and_test(args, start_time)
    elif args.action == 'infer':
        infer(args, start_time)
    else:
        assert False
# fmt: on
