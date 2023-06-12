#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
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
def get_df(smpl, result, smpldict):
    df = pd.DataFrame(
        {
            "Predicted": np.array([row[0].value for row in result]),
            "Truth": np.array(
                [bresp.xscale for bresp in smpldict[smpl]["birth-responses"]], dtype=str
            ),
        }
    )
    utils.make_dl_plot(smpl, df, args.outdir)
    df.to_csv(csvfn(smpl))
    return df


# ----------------------------------------------------------------------------------------
def read_plot_csv():
    for smpl in ["train", "test"]:
        df = pd.read_csv(csvfn(smpl))
        utils.make_dl_plot(smpl, df, args.outdir)


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

    n_trees = len(samples["trees"])
    idxs, smpldict = {}, {}
    if args.test_xscale_values is None:
        idxs["train"] = random.sample(range(n_trees), round(args.train_frac * n_trees))
        idxs["test"] = [i for i in range(n_trees) if i not in idxs["train"]]
    else:
        n_test = round((1. - args.train_frac) * n_trees)
        avail_indices = [i for i, r in enumerate(samples['birth-responses']) if r._param_dict['xscale'] in args.test_xscale_values]  # indices of all trees with the specified xscale value
        idxs["test"] = random.sample(avail_indices, n_test)  # note that this'll change the distribution of xscale values in the training sample (so make sure that n_test isn't a large fraction of the trees with each xscale value
        idxs["train"] = [i for i in range(n_trees) if i not in idxs["test"]]
        all_vals = [r._param_dict['xscale'] for r in samples['birth-responses']]
        val_counts = {v : all_vals.count(v) for v in set(all_vals)}
        n_remaining = sum(val_counts[v] for v in args.test_xscale_values) - n_test
        print('    --test-xscale-value: chose %d test samples with xscale values among %s (leaving %d with those values) from %d total samples with xscale value counts: %s' % (n_test, ' '.join('%.2f'%v for v in args.test_xscale_values), n_remaining, n_trees, '   '.join('%.2f %d' % (v, c) for v, c in sorted(val_counts.items(), key=operator.itemgetter(0)))))

    for smpl in ["train", "test"]:
        smpldict[smpl] = {
            key: [val[i] for i in idxs[smpl]] for key, val in samples.items()
        }

    param_to_predict = [
        [ConstantResponse(birth_resp.xscale)]
        for birth_resp in smpldict["train"]["birth-responses"]
    ]

    model = NeuralNetworkModel(
        smpldict["train"]["trees"], param_to_predict, network_layers=args.model_size
    )
    model.fit(epochs=args.epochs)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    print("  writing train/test results to %s" % args.outdir)

    result = model.predict(smpldict["train"]["trees"])
    get_df("train", result, smpldict)
    result = model.predict(smpldict["test"]["trees"])
    get_df("test", result, smpldict)

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
parser.add_argument("--test-xscale-values", type=float, nargs='+', help='if set, choose test samples only from among those with this (birth) xscale value.')
parser.add_argument("--model-size", default="tiny", choices=["small", "tiny", None])
parser.add_argument(
    "--test",
    action="store_true",
    help="sets things to be super fast, so not useful for real inference, but just to check if things are running properly",
)
parser.add_argument("--random-seed", default=0, type=int, help="random seed")
parser.add_argument("--overwrite", action="store_true")

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
