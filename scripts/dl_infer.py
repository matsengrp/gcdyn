#!/usr/bin/env python
import numpy as np
import argparse
import os
import sys
from sklearn import preprocessing
import joblib
import dill
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
    if args.dont_scale_params:
        return copy.copy(in_pvals), scaler
    if scaler is None:
        scaler = preprocessing.StandardScaler().fit(in_pvals)  # scaler = preprocessing.MinMaxScaler(feature_range=(0, 10)).fit(in_pvals)
    sc_pvals = scaler.inverse_transform(in_pvals) if inverse else scaler.transform(in_pvals)
    if debug:
        if debug:  # and smpl == smplist[0]:
            print("    %sscaling %d variables: %s" % ("reverse " if inverse else "", len(args.params_to_predict), args.params_to_predict,))
            print("                                  before                             after")
            print("                           mean   var     min   max         mean   var     min   max")
        print_debug(in_pvals, sc_pvals)
    return sc_pvals, scaler


# ----------------------------------------------------------------------------------------
def collapse_bundles(args, resps, sstats):
    resps = [
        NeuralNetworkModel._collapse_identical_list(resps[i : i + args.dl_bundle_size])
        for i in range(0, len(resps), args.dl_bundle_size)
    ]
    sstats = [
        {
            tkey: (min if tkey == "tree" else np.mean)(
                [float(sstats[i + j][tkey]) for j in range(args.dl_bundle_size)]
            )
            for tkey in sstats[i]
        }
        for i in range(0, len(sstats), args.dl_bundle_size)
    ]  # mean of each summary stat over trees in each bundle ('tree' key is an index, so take min/index of first one)
    return resps, sstats

# ----------------------------------------------------------------------------------------
def write_prediction(args, punscaled, true_resps=None, true_sstats=None, smpl=None):
    dfdata = {  # make empty df
        "%s-%s" % (param, ptype): []
        for param in args.params_to_predict
        for ptype in (["predicted"] if true_resps is None else ["truth", "predicted"])
    }
    dfdata['tree-index'] = []
    assert true_resps is None or len(punscaled) == len(true_resps)
    for itr, prlist in enumerate(punscaled):
        dfdata['tree-index'].append(true_sstats[itr]['tree'])
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
    pvals = [[float(resp.value) for resp in plist] for plist in pred_resps]
    punscaled, _ = scale_vals(args, pvals, smpl=smpl, scaler=scaler, inverse=True)  # *un* scale according to the training scaling (if scaler is not None, it should be the training scaler)
    true_resps, true_sstats = None, None
    if args.is_simu:
        true_resps, true_sstats = [spld[tk] for tk in ["birth-responses", "sstats"]]
        if args.dl_bundle_size > 1:
            true_resps, true_sstats = collapse_bundles(args, true_resps, true_sstats)
        assert len(pred_resps) == len(true_resps)
    df = write_prediction(args, punscaled, true_resps=true_resps, true_sstats=true_sstats, smpl=smpl)
    return df

# ----------------------------------------------------------------------------------------
def plot_existing_results(args):
    prdfs, smpldict = {}, {}
    for smpl in ["train", "test"]:
        prdfs[smpl] = pd.read_csv(csvfn(args, smpl))
    seqmeta = read_meta_csv(args.indir)
    utils.make_dl_plots(
        prdfs,
        seqmeta,
        args.params_to_predict,
        args.outdir + "/plots",
        is_simu=args.is_simu,
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
def read_meta_csv(mdir):
    metafos = []
    with open('%s/meta.csv'%mdir) as lmfile:
        reader = csv.DictReader(lmfile)
        for line in reader:
            metafos.append(line)
    return metafos

# ----------------------------------------------------------------------------------------
def read_tree_files(args):
    # ----------------------------------------------------------------------------------------
    def check_bundles(samples, debug=False):
        rnames = ['birth', 'death']
        ibund, broken_bundles = -1, []
        bundle_rates, tree_indices = {n : [] for n in rnames}, []
        for iresp, (br, dr) in enumerate(zip(samples["birth-responses"], samples["death-responses"])):
            if iresp % args.dl_bundle_size == 0:
                for rn in rnames:
                    if len(bundle_rates[rn]) > 1:
                        broken_bundles.append(ibund)
                        print('    %s bundle with index %d (tree indices %d to %d) has multiple %s rates:\n      %s' % (utils.color("yellow", "warning"), ibund, min(tree_indices), max(tree_indices), rn, '\n      '.join(str(r) for r in bundle_rates[rn])))
                        # print(' '.join(str(i) for i in tree_indices))  # maybe want to print all broken tree indices for deletion by hand
                bundle_rates['birth'], bundle_rates['death'] = [br], [dr]
                tree_indices = []
                ibund += 1
                if debug:
                    print('  bundle %d (size %d)' % (ibund, args.dl_bundle_size))
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
    samples = {}
    rfn, tfn, sfn = [
        "%s/%s" % (args.indir, s)
        for s in ["responses.pkl", "encoded-trees.npy", "summary-stats.csv"]
    ]
    rstr = ''
    if args.is_simu:
        with open(rfn, "rb") as rfile:
            pklfo = pickle.load(rfile)
        rstr = ' (%d responses from %s)' % (len(pklfo), rfn)
        for tk in ["birth", "death"]:
            samples[tk + "-responses"] = [tfo[tk] for tfo in pklfo]
    samples["trees"] = encode.read_trees(tfn)
    samples["sstats"] = []
    with open(sfn) as sfile:
        reader = csv.DictReader(sfile)
        for line in reader:
            samples["sstats"].append(line)
    print("    read %d trees from %s%s" % (len(samples["trees"]), tfn, rstr))
    if args.is_simu:
        print("      first response pair:\n        birth: %s\n        death: %s" % (samples["birth-responses"][0], samples["death-responses"][0]))
        check_bundles(samples)
    if len(samples["trees"]) % args.dl_bundle_size != 0:
        if args.discard_extra_trees:
            n_remain = len(samples["trees"]) % args.dl_bundle_size
            print('  --discard-extra-trees: discarding %d trees from end of input since total N trees %d isn\'t evenly divisible by bundle size %d' % (n_remain, len(samples['trees']), args.dl_bundle_size))
            for tk in samples:
                samples[tk] = samples[tk][: len(samples[tk]) - n_remain]
        else:
            raise Exception('N trees %d not divisible by bundle size %d' % (len(samples["trees"]), args.dl_bundle_size))
    return samples


# ----------------------------------------------------------------------------------------
def predict_and_plot(args, model, smpldict, scaler, smpls):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    prdfs = {}
    for smpl in smpls:
        prdfs[smpl] = get_prediction(args, model, smpldict[smpl], scaler, smpl=smpl)
    seqmeta = read_meta_csv(args.indir)
    utils.make_dl_plots(
        prdfs,
        seqmeta,
        args.params_to_predict,
        args.outdir + "/plots",
        is_simu=args.is_simu,
        validation_split=0 if smpl=='infer' else args.validation_split,
    )

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
    model = NeuralNetworkModel(responses[0], bundle_size=args.dl_bundle_size)
    model.build_model(
        smpldict["train"]["trees"],
        responses,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        ema_momentum=args.ema_momentum,
        prebundle_layer_cfg=args.prebundle_layer_cfg,
    )
    model.fit(epochs=args.epochs, validation_split=args.validation_split)
    model.network.save(encode.output_fn(args.outdir, 'model', None))
    joblib.dump(scalers['train'], encode.output_fn(args.outdir, 'train-scaler', None))
    with open(encode.output_fn(args.outdir, 'example-responses', None), 'wb') as dfile:
        dill.dump(responses[0], dfile)  # dump list of response fcns for one tree (so that when reading the model to infer, we can tell the neural network the structure of the responses)

    predict_and_plot(args, model, smpldict, scalers["train"], ['train', 'test'])
    print("    total dl inference time: %.1f sec" % (time.time() - start_time))

# ----------------------------------------------------------------------------------------
def read_model_files(args, samples):
    erfn = encode.output_fn(args.model_dir, 'example-responses', None)
    if os.path.exists(erfn):
        print('    reading example responses from %s' % erfn)
        with open(erfn, "rb") as rfile:
            example_response_list = pickle.load(rfile)
    else:
        print('  %s example responses file %s doesn\'t exist, so trying to get from input info (probably ok on simulation)' % (utils.color('yellow', 'warning'), erfn))
        pvals = get_pvlists(args, samples)  # only really need the pvals to get the scaler (and maybe we should anyway be using the scaler from the training sample?)
        pscaled, inf_scaler = scale_vals(args, pvals)
        example_response_list = [ConstantResponse(v) for v in pscaled[0]]  # well, we used pscaled for the example response, but this is a really hacky way of telling the neural network how many parameters to expect
    scfn = encode.output_fn(args.model_dir, 'train-scaler', None)
    if os.path.exists(scfn):
        print('    reading training scaler from %s' % scfn)
        use_scaler = joblib.load(scfn)
    else:
        print('  %s training scaler file %s doesn\'t exist, so fitting new scaler on inference sample (which isn\'t correct, but may be ok)' % (utils.color('yellow', 'warning'), scfn))
        use_scaler = inf_scaler

    model = NeuralNetworkModel(example_response_list, bundle_size=args.dl_bundle_size)
    model.load(encode.output_fn(args.model_dir, 'model', None))

    return use_scaler, model

# ----------------------------------------------------------------------------------------
def infer(args, start_time):
    smpldict = {'infer' : read_tree_files(args)}
    scaler, model = read_model_files(args, smpldict['infer'])
    predict_and_plot(args, model, smpldict, scaler, ['infer'])
    print("    total dl inference time: %.1f sec" % (time.time() - start_time))

# ----------------------------------------------------------------------------------------
def get_parser():
    helpstr = """
    Infer affinity response function on gcdyn simulation using deep learning neural networks.
    Two actions: <train> trains and tests a new dl model on the sample in the input dir, whereas
    <infer> infers parameters with an existing dl model.
    Example usage:
        gcd-dl train --indir <input dir> --outdir <output dir>
        gcd-dl infer --indir <input dir> --outdir <output dir> --model-dir <model dir>
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
    parser.add_argument("--model-dir", help="file with saved deep learning model for inference")
    parser.add_argument("--outdir", required=True, help="output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dl-bundle-size", type=int, default=50, help='\'dl-\' is to differentiate from \'simu-\' bundle size when calling this from cf-gcdyn.py')
    parser.add_argument("--discard-extra-trees", action="store_true", help='By default, the number of trees during inference must be evenly divisible by --dl-bundle-size. If this is set, however, any extras are discarded to allow inference.')
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
    if args.action == 'infer' and args.model_dir is None:
        raise Exception('must specify --model-dir for \'infer\' action')
    if args.test:
        args.epochs = 10
    if args.action == 'train' and not args.is_simu:
        raise Exception('need to set --is-simu when training')

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    import tensorflow as tf  # this is super slow, don't want to wait for this to get help message

    tf.keras.utils.set_random_seed(args.random_seed)

    if args.action == 'train':
        if os.path.exists(csvfn(args, "test")) and not args.overwrite:
            print("    csv files already exist, so just replotting (override with --overwrite): %s" % csvfn(args, "test"))
            plot_existing_results(args)
            sys.exit(0)
        train_and_test(args, start_time)
    elif args.action == 'infer':
        infer(args, start_time)
    else:
        assert False
# fmt: on
