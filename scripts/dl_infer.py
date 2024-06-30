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
from gcdyn.nn import ParamNetworkModel, PerCellNetworkModel
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
        assert args.model_type in ['sigmoid', 'per-cell']
        for ivar, vname in enumerate(svar_list):
            for dstr, pvs in zip(("before", "after"), (pvals_before, pvals_scaled)):
                bstr = "   " if dstr != "before" else "      %10s %7s" % (vname, smpl)
                print("%s %s%s %s%s" % (bstr, fnstr(pvs, np.mean), fnstr(pvs, np.var), fnstr(pvs, min), fnstr(pvs, max), ), end="" if dstr == "before" else "\n")
    # ----------------------------------------------------------------------------------------
    svar_list = args.params_to_predict if args.model_type=='sigmoid' else ['fitness']
    if args.dont_scale_params:
        return copy.copy(in_pvals), scaler
    if inverse:
        assert scaler is not None
    if scaler is None:
        scaler = preprocessing.StandardScaler().fit(in_pvals)  # scaler = preprocessing.MinMaxScaler(feature_range=(0, 10)).fit(in_pvals)
    sc_pvals = scaler.inverse_transform(in_pvals) if inverse else scaler.transform(in_pvals)
    if debug:
        if debug:  # and smpl == smplist[0]:
            print("    %sscaling %d variables: %s" % ("reverse " if inverse else "", len(svar_list), svar_list,))
            print("                                  before                             after")
            print("                           mean   var     min   max         mean   var     min   max")
        print_debug(in_pvals, sc_pvals)
    return sc_pvals, scaler


# ----------------------------------------------------------------------------------------
def collapse_bundles(args, resps, sstats):
    resps = [
        ParamNetworkModel._collapse_identical_list(resps[i : i + args.dl_bundle_size])
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
def write_sigmoid_prediction(args, punscaled, true_resps=None, true_sstats=None, smpl=None):
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
def write_per_cell_prediction(args, pred_fitnesses, enc_trees, true_fitnesses=None, true_resps=None, true_sstats=None, smpl=None):
    assert true_fitnesses is None or len(pred_fitnesses) == len(true_fitnesses)
    dfdata = {  # make empty df
        "fitness-%s"%ptype : []
        for ptype in (["predicted"] if true_fitnesses is None else ["truth", "predicted"])
    }
    dfdata['tree-index'] = []
    dfdata['phenotype'] = []
    for param in args.params_to_predict:
        dfdata['%s-truth'%param] = []
    for itr, efit in enumerate(pred_fitnesses):
        pred_ndicts = encode.decode_matrices(enc_trees[itr], efit)
        if true_fitnesses is not None:
            true_ndicts = encode.decode_matrices(enc_trees[itr], true_fitnesses[itr])
            assert len(pred_ndicts) == len(true_ndicts)
            for icell, (pdict, tdict) in enumerate(zip(pred_ndicts, true_ndicts)):
                pdict['fitness-predicted'] = pdict['fitness']
                del pdict['fitness']
                pdict['fitness-truth'] = tdict['fitness']
                for ip, param in enumerate(args.params_to_predict):  # writes true parameter values for each cell, which kind of sucks, but they're only nonzero for the first cell in each tree
                    dfdata["%s-truth" % param].append(get_pval(param, true_resps[itr], true_sstats[itr]) if icell==0 else 0)
        for ndict in pred_ndicts:
            dfdata['tree-index'].append(true_sstats[itr]['tree'])  # NOTE tree-index isn't necessarily equal to <itr>
            for tk in [k for k in ndict if k in dfdata]:
                dfdata[tk].append(ndict[tk])
    df = pd.DataFrame(dfdata)
    print("  writing %s results to %s" % (smpl, args.outdir))
    df.to_csv(csvfn(args, smpl))
    return df

# ----------------------------------------------------------------------------------------
def get_prediction(args, model, spld, scaler, smpl=None):
    true_fitnesses, true_resps, true_sstats = None, None, None
    if args.model_type == 'sigmoid':
        pred_resps = model.predict(spld["trees"])  # note that this returns constant response fcns that are just holders for the predicted values (i.e. don't directly relate to true/input response fcns)
        pvals = [[float(resp.value) for resp in plist] for plist in pred_resps]  # order corresponds to args.params_to_predict
        punscaled, _ = scale_vals(args, pvals, smpl=smpl, scaler=scaler, inverse=True)  # *un* scale according to the training scaling (if scaler is not None, it should be the training scaler)
        if args.is_simu:
            true_resps, true_sstats = [spld[tk] for tk in ["birth-responses", "sstats"]]
            if args.dl_bundle_size > 1:
                true_resps, true_sstats = collapse_bundles(args, true_resps, true_sstats)
            assert len(pred_resps) == len(true_resps)
        df = write_sigmoid_prediction(args, punscaled, true_resps=true_resps, true_sstats=true_sstats, smpl=smpl)
    elif args.model_type == 'per-cell':
        assert args.dl_bundle_size == 1
        pred_fitnesses = model.predict(spld["trees"]).numpy()
        # print(pred_fitnesses[0])
        for pfit, etree in zip(pred_fitnesses, spld['trees']):
            encode.reset_fill_entries(pfit, etree)
        punscaled, _ = scale_vals(args, extract_fitnesses(pred_fitnesses), smpl=smpl, scaler=scaler, inverse=True)  # *un* scale according to the training scaling (if scaler is not None, it should be the training scaler)
        unscaled_fitnesses = encode.matricize_fitnesses(punscaled, pred_fitnesses)
        if args.is_simu:
            true_fitnesses, true_resps, true_sstats = [spld[tk] for tk in ["fitnesses", "birth-responses", "sstats"]]
            # print(true_fitnesses[0])
            # assert False
        df = write_per_cell_prediction(args, unscaled_fitnesses, spld["trees"], true_fitnesses=true_fitnesses, true_resps=true_resps, true_sstats=true_sstats, smpl=smpl)
    else:
        assert False
    return df

# ----------------------------------------------------------------------------------------
def plot_existing_results(args):
    prdfs, smpldict = {}, {}
    for smpl in ["train", "test"]:
        prdfs[smpl] = pd.read_csv(csvfn(args, smpl))
    seqmeta = read_meta_csv(args.indir)
    utils.make_dl_plots(
        args.model_type,
        prdfs,
        seqmeta,
        args.params_to_predict,
        args.outdir + "/plots",
        is_simu=args.is_simu,
        validation_split=args.validation_split,
        trivial_encoding=args.use_trivial_encoding,
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
    if args.model_type == 'sigmoid':
        pvals = [
            [get_pval(pname, bresp, sts) for pname in args.params_to_predict]
            for bresp, sts in zip(
                samples["birth-responses"], samples["sstats"]
            )
        ]
    elif args.model_type == 'per-cell':
        pvals = extract_fitnesses(samples['fitnesses'])
    else:
        assert False
    return pvals

# ----------------------------------------------------------------------------------------
def extract_fitnesses(matrix_list):
    """Convert encoded fitness matrix into list of (length-1 lists) with fitness, for input to scale_vals"""
    return [[nfo['fitness']] for fmtx in matrix_list for nfo in encode.decode_matrix('fitness', fmtx)]

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
    rfn, tfn, ffn, sfn = [
        "%s/%s" % (args.indir, s)
        for s in ["responses.pkl", "encoded-trees.npy", "encoded-fitnesses.npy", "summary-stats.csv"]
    ]
    rstr = ''
    if args.is_simu:
        with open(rfn, "rb") as rfile:
            pklfo = pickle.load(rfile)
        rstr = ' (%d responses from %s)' % (len(pklfo), rfn)
        for tk in ["birth", "death"]:
            samples[tk + "-responses"] = [tfo[tk] for tfo in pklfo]
    samples["trees"] = encode.read_trees(tfn)
    samples["fitnesses"] = encode.read_trees(ffn)
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
def predict_and_plot(args, model, smpldict, smpls, scaler=None):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    prdfs = {}
    for smpl in smpls:
        prdfs[smpl] = get_prediction(args, model, smpldict[smpl], scaler, smpl=smpl)
    seqmeta = read_meta_csv(args.indir)
    utils.make_dl_plots(
        args.model_type,
        prdfs,
        seqmeta,
        args.params_to_predict,
        args.outdir + "/plots",
        is_simu=args.is_simu,
        validation_split=0 if smpl=='infer' else args.validation_split,
        trivial_encoding=args.use_trivial_encoding,
    )

# ----------------------------------------------------------------------------------------
def train_and_test(args, start_time):
    # ----------------------------------------------------------------------------------------
    def train_sigmoid(smpldict, pscaled, max_leaf_count):
        responses = [[ConstantResponse(v) for v in vlist] for vlist in pscaled["train"]]  # order corresponds to args.params_to_predict (constant response is just a container for one value, and note we don't bother to set the name)
        model = ParamNetworkModel(responses[0], bundle_size=args.dl_bundle_size)
        assert args.params_to_predict == utils.sigmoid_params
        model.build_model(max_leaf_count, dropout_rate=args.dropout_rate, learning_rate=args.learning_rate, ema_momentum=args.ema_momentum, prebundle_layer_cfg=args.prebundle_layer_cfg)
        model.fit(smpldict["train"]["trees"], responses, epochs=args.epochs, validation_split=args.validation_split)
        with open(encode.output_fn(args.outdir, 'example-responses', None), 'wb') as dfile:
            dill.dump(responses[0], dfile)  # dump list of response fcns for one tree (so that when reading the model to infer, we can tell the neural network the structure of the responses)
        return model
    # ----------------------------------------------------------------------------------------
    def train_per_cell(smpldict, pscaled, max_leaf_count):
        # seqmeta = read_meta_csv(args.indir)
        # affy_vals = [float(m['affinity']) for m in seqmeta]
        model = PerCellNetworkModel()
        model.build_model(max_leaf_count, dropout_rate=args.dropout_rate, learning_rate=args.learning_rate, ema_momentum=args.ema_momentum)
        new_fitnesses = encode.matricize_fitnesses(pscaled['train'], smpldict['train']['fitnesses'])
        assert len(smpldict['train']['fitnesses']) == len(new_fitnesses)
        model.fit(smpldict["train"]["trees"], new_fitnesses, epochs=args.epochs, validation_split=args.validation_split)
        return model
    # ----------------------------------------------------------------------------------------
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

    leaf_counts = set([len(t[0]) for t in smpldict["train"]["trees"]])  # length of first row in encoded tree (i guess really it'd be better to also include test trees in this, but in practice it probably doesn't matter)
    if len(leaf_counts) != 1:
        raise Exception("encoded trees have different lengths: %s" % " ".join(str(c) for c in leaf_counts))
    max_leaf_count = list(leaf_counts)[0]

    # handle various scaling/re-encoding stuff
    pscaled, scalers = {}, {}  # scaled parameters and scalers
    for smpl in smplist:
        pvals = get_pvlists(args, smpldict[smpl])
        pscaled[smpl], scalers[smpl] = scale_vals(args, pvals, smpl=smpl)
    joblib.dump(scalers['train'], encode.output_fn(args.outdir, 'train-scaler', None))

    # silly encodings for testing that essentially train on the output values
    if args.use_trivial_encoding:
        for smpl in smplist:
            predict_vals = pscaled[smpl]
            if args.model_type == 'per-cell':
                predict_vals = encode.matricize_fitnesses(predict_vals, smpldict[smpl]['fitnesses'])
            encode.trivialize_encodings(smpldict[smpl]["trees"], args.model_type, predict_vals, noise=False, n_debug=3)

    if args.model_type == 'sigmoid':
        model = train_sigmoid(smpldict, pscaled, max_leaf_count)
    elif args.model_type == 'per-cell':
        model = train_per_cell(smpldict, pscaled, max_leaf_count)
    else:
        assert False
    model.network.save(encode.output_fn(args.outdir, 'model', None))

    predict_and_plot(args, model, smpldict, ['train', 'test'], scaler=scalers["train"])
    print("    total dl inference time: %.1f sec" % (time.time() - start_time))

# ----------------------------------------------------------------------------------------
def read_model_files(args, samples):
    # ----------------------------------------------------------------------------------------
    def read_example_response():
        erfn = encode.output_fn(args.model_dir, 'example-responses', None)
        if os.path.exists(erfn):
            print('    reading example responses from %s' % erfn)
            with open(erfn, "rb") as rfile:
                example_response_list = pickle.load(rfile)
        else:
            print('  %s example responses file %s doesn\'t exist, so trying to get from input info (probably ok on simulation)' % (utils.color('yellow', 'warning'), erfn))
            pvals = get_pvlists(args, samples)  # only need the pvals to get the scaler for example_response_list, which doesn't use the actual parameters values, so this is pretty silly (e.g. maybe I could just replace with example_response_list = [ConstantResponse(0) for _ in args.params_to_predict]?)
            pscaled, inf_scaler = scale_vals(args, pvals)
            example_response_list = [ConstantResponse(v) for v in pscaled[0]]  # well, we used pscaled for the example response, but this is a really hacky way of telling the neural network how many parameters to expect
        return example_response_list
    # ----------------------------------------------------------------------------------------
    def read_scaler():
        scfn = encode.output_fn(args.model_dir, 'train-scaler', None)
        if os.path.exists(scfn):
            print('    reading training scaler from %s' % scfn)
            use_scaler = joblib.load(scfn)
        else:
            assert False  # shouldn't need this any more (?) [would need to get inf_scaler from fcn above now that I've rearranged things]
            print('  %s training scaler file %s doesn\'t exist, so fitting new scaler on inference sample (which isn\'t correct, but may be ok)' % (utils.color('yellow', 'warning'), scfn))
            use_scaler = inf_scaler
        return use_scaler
    # ----------------------------------------------------------------------------------------
    use_scaler = read_scaler()
    if args.model_type == 'sigmoid':
        model = ParamNetworkModel(read_example_response(), bundle_size=args.dl_bundle_size)
    elif args.model_type == 'per-cell':
        model = PerCellNetworkModel()
    else:
        assert False

    model.load(encode.output_fn(args.model_dir, 'model', None))

    return use_scaler, model

# ----------------------------------------------------------------------------------------
def infer(args, start_time):
    smpldict = {'infer' : read_tree_files(args)}
    scaler, model = read_model_files(args, smpldict['infer'])
    predict_and_plot(args, model, smpldict, ['infer'], scaler=scaler)
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
    parser.add_argument("--model-type", choices=['sigmoid', 'per-cell'], default='per-cell', help='type of neural network model, sigmoid: infer 3 params of sigmoid fcn, per-cell: infer fitness of each individual cell')
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dl-bundle-size", type=int, default=1, help='\'dl-\' is to differentiate from \'simu-\' bundle size when calling this from cf-gcdyn.py')
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
    if args.use_trivial_encoding and not args.dont_scale_params:
        print('  %s --use-trivial-encoding: turning on --dont-scale-params since parameter scaling needs fixing to work with trivial encoding' % utils.wrnstr())
        args.dont_scale_params = True

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
