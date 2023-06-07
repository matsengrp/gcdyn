#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# import colored_traceback.always  # need to add this to installation stuff, i'm not sure how to do it atm
import time
import pickle
import pandas as pd
import seaborn as sns
import random


# ----------------------------------------------------------------------------------------
def csvfn(smpl):
    return "%s/%s.csv" % (args.outdir, smpl)


# ----------------------------------------------------------------------------------------
def make_plot(smpl, df):
    plt.clf()
    sns.set_palette("viridis", 8)
    hord = sorted(set(df["Truth"]))
    # sns.histplot(df, x="Predicted", hue="Truth", hue_order=hord, palette="tab10", bins=30, multiple="stack", ).set(title=smpl)
    ax = sns.boxplot(df, x='Truth', y='Predicted', boxprops={'facecolor':'None'})
    if len(df) < 2000:
        ax = sns.swarmplot(df, x='Truth', y='Predicted', size=4, alpha=0.6)
    ax.set(title=smpl)
    for xv, xvl in zip(ax.get_xticks(), ax.get_xticklabels()):
        plt.plot([xv - 0.5, xv + 0.5], [float(xvl._text), float(xvl._text)], color='darkred', linestyle='--', linewidth=3, alpha=0.7)
    # sns.scatterplot(df, x='Truth', y='Predicted')
    # xvals, yvals = df['Truth'], df['Predicted']
    # plt.plot([0.95 * min(xvals), 1.05 * max(xvals)], [0.95 * min(yvals), 1.05 * max(yvals)], color='darkred', linestyle='--', linewidth=3, alpha=0.7)
    plt.savefig("%s/%s-hist.svg" % (args.outdir, smpl))
    df.to_csv(csvfn(smpl))


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
    make_plot(smpl, df)
    return df


# ----------------------------------------------------------------------------------------
def read_plot_csv():
    for smpl in ["train", "test"]:
        df = pd.read_csv(csvfn(smpl))
        make_plot(smpl, df)


# ----------------------------------------------------------------------------------------
def train_and_test():
    from gcdyn.models import NeuralNetworkModel
    from gcdyn.poisson import ConstantResponse

    with open(args.response_file, "rb") as rfile:
        pklfo = pickle.load(rfile)
    samples = {
        k + "-responses": [tfo[k] for tfo in pklfo]
        for k in ["birth", "death"]
    }
    samples["trees"] = np.load(args.tree_file)
    n_leaf_list = ([len(t[0]) for t in samples['trees']])
    max_leaf_count = max(args.min_n_max_leaves, max(n_leaf_list))  # model complains if this is 70, i'm not sure why but whatever
    print('    padding encoded trees to max leaf count %d (all leaf counts: %s)' % (max_leaf_count, ' '.join(str(c) for c in set(n_leaf_list))))
    padded_trees = []
    for itree, etree in enumerate(samples['trees']):
        assert len(etree) == 4  # make sure there's 4 rows
        assert len(set(len(r) for r in etree)) == 1  # and that every row is the same length
        padded_trees.append(np.pad(etree, ((0, 0), (0, max_leaf_count - len(etree[0])))))
        # if itree == 0:
        #     before_len = len(etree[0])
        #     np.set_printoptions(precision=3, suppress=True, linewidth=99999)
        #     print('  padded length from %d to %d' % (before_len, len(etree[0])))
        #     print(etree)
    samples['trees'] = np.array(padded_trees)
    print("    read %d trees from %s (%d responses from %s)" % (len(samples["trees"]), args.tree_file, len(pklfo), args.response_file))
    print(
        "      first response pair:\n        birth: %s\n        death: %s"
        % (samples["birth-responses"][0], samples["death-responses"][0])
    )

    n_trees = len(samples["trees"])
    idxs, smpldict = {}, {}
    idxs["train"] = random.sample(range(n_trees), int(args.train_frac * n_trees))
    idxs["test"] = [i for i in range(n_trees) if i not in idxs["train"]]

    for smpl in ["train", "test"]:
        smpldict[smpl] = {
            key: [val[i] for i in idxs[smpl]] for key, val in samples.items()
        }

    param_to_predict = [
        [ConstantResponse(birth_resp.xscale)]
        for birth_resp in smpldict["train"]["birth-responses"]
    ]

    model = NeuralNetworkModel(
        None, param_to_predict, network_layers=args.model_size, encoded_trees=smpldict["train"]["trees"], max_leaf_count=max_leaf_count
    )
    model.fit(epochs=args.epochs)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    print("  writing train/test results to %s" % args.outdir)

    result = model.predict(None, encoded_trees=smpldict["train"]["trees"])
    get_df("train", result, smpldict)
    result = model.predict(None, encoded_trees=smpldict["test"]["trees"])
    get_df("test", result, smpldict)

    print("    total dl inference time: %.1f sec" % (time.time() - start))


# ----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tree-file", required=True, help="input file with list of encoded trees in numpy npy format")
parser.add_argument("--response-file", required=True, help="input file with list of response functions in pickle format")
parser.add_argument("--outdir", required=True, help='output directory')
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--min-n-max-leaves", type=int, default=200)
parser.add_argument(
    "--train-frac", type=float, default=0.8, help="train on this fraction of the trees"
)
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
