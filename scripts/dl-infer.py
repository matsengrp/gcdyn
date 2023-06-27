#!/usr/bin/env python
import numpy as np
import argparse
import os
import sys
import operator

# import colored_traceback.always  # need to add this to installation stuff, i'm not sure how to do it atm
import time
import pickle
import pandas as pd
import random

from gcdyn import utils


# ----------------------------------------------------------------------------------------
def csvfn(smpl):
    return "%s/%s.csv" % (args.outdir, smpl)


# ----------------------------------------------------------------------------------------
def get_prediction(smpl, model, smpldict):
    pred_resps = model.predict(smpldict[smpl]["trees"])
    true_resps = smpldict[smpl]["birth-responses"]
    dfdata = {'%s-%s'%(param, ptype) : [] for param in args.params_to_predict for ptype in ['truth', 'predicted']}
    assert len(pred_resps) == len(true_resps)
    for tr_resp, prlist in zip(true_resps, pred_resps):
        for ip, param in enumerate(args.params_to_predict):
            dfdata['%s-truth'%param].append(getattr(tr_resp, param))
            dfdata['%s-predicted'%param].append(float(prlist[ip].value))
    df = pd.DataFrame(dfdata)
    df.to_csv(csvfn(smpl))
    return df


# ----------------------------------------------------------------------------------------
def read_plot_csv():
    prdfs = {}
    for smpl in ["train", "test"]:
        prdfs[smpl] = pd.read_csv(csvfn(smpl))
    utils.make_dl_plots(prdfs, args.params_to_predict, args.outdir)


# ----------------------------------------------------------------------------------------
def get_traintest_indices(samples):
    def is_avail(resp):
        for pname in args.test_param_vals:
            if resp._param_dict[pname] not in args.test_param_vals[pname]:
                return False  # if this response's e.g. xscale value isn't in the allowed values from the command line, skip it
        return True

    def print_stats():
        for pname in args.test_param_vals: #samples["bith-responses"][0]._param_dict:
            all_vals = [r._param_dict[pname] for r in samples["birth-responses"]]
            val_counts = {v: all_vals.count(v) for v in set(all_vals)}
            n_remaining = sum(val_counts.get(v, 0) for v in args.test_param_vals[pname]) - n_test
            print('      --test-%s-values: restricted to %d / %d %s values (%s), choosing %d / %d with these values from original value counts: %s' % (
                pname,
                len(args.test_param_vals[pname]),
                len(set(all_vals)),
                pname,
                " ".join("%.2f" % v for v in args.test_param_vals[pname]),
                n_test,
                n_remaining + n_test,
                "   ".join(
                    "%.2f %d" % (v, c)
                    for v, c in sorted(val_counts.items(), key=operator.itemgetter(0))
                ),
            ))

    n_trees = len(samples["trees"])
    n_test = round((1.0 - args.train_frac) * n_trees)
    idxs = {}
    if args.test_param_vals is None:
        idxs["train"] = random.sample(range(n_trees), round(args.train_frac * n_trees))
        idxs["test"] = [i for i in range(n_trees) if i not in idxs["train"]]
    else:
        avail_indices = [
            i
            for i, r in enumerate(samples["birth-responses"])
            if is_avail(r)
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

    with open(args.response_file, "rb") as rfile:
        pklfo = pickle.load(rfile)
    samples = {k + "-responses": [tfo[k] for tfo in pklfo] for k in ["birth", "death"]}
    samples["trees"] = encode.read_trees(args.tree_file)
    print(
        "    read %d trees from %s (%d responses from %s)"
        % (len(samples["trees"]), args.tree_file, len(pklfo), args.response_file)
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
    print('      N trees: %d train   %d test' % (len(smpldict['train']['trees']), len(smpldict['test']['trees'])))

    pred_resps = [
        [ConstantResponse(getattr(birth_resp, p)) for p in args.params_to_predict]
        for birth_resp in smpldict["train"]["birth-responses"]
    ]

    model = NeuralNetworkModel(
        smpldict["train"]["trees"], pred_resps, network_layers=args.model_size
    )
    model.fit(epochs=args.epochs)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    print("  writing train/test results to %s" % args.outdir)

    prdfs = {}
    for smpl in ['train', 'test']:
        prdfs[smpl] = get_prediction(smpl, model, smpldict)
    utils.make_dl_plots(prdfs, args.params_to_predict, args.outdir)

    print("    total dl inference time: %.1f sec" % (time.time() - start))


# ----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--tree-file",
    required=True,
    help="input file with list of encoded trees in numpy npy format",
)
parser.add_argument(
    "--response-file",
    required=True,
    help="input file with list of response functions in pickle format",
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
parser.add_argument("--model-size", default="tiny", choices=["small", "tiny", None], help="Parameters from the birth model that we should try to predict.")
parser.add_argument("--params-to-predict", default=["xscale", 'shift'], nargs='+', choices=["xscale", "xshift"])
parser.add_argument(
    "--test",
    action="store_true",
    help="sets things to be super fast, so not useful for real inference, but just to check if things are running properly",
)
parser.add_argument("--random-seed", default=0, type=int, help="random seed")
parser.add_argument("--overwrite", action="store_true")

start = time.time()
args = parser.parse_args()
args.test_param_vals = None
if args.test_xscale_values is not None or args.test_xshift_values is not None:
    args.test_param_vals = {}
    if args.test_xscale_values is not None:
        args.test_param_vals.update({'xscale' : args.test_xscale_values})
        delattr(args, 'test_xscale_values')
    if args.test_xshift_values is not None:
        args.test_param_vals.update({'xshift' : args.test_xshift_values})
        delattr(args, 'test_xshift_values')
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
