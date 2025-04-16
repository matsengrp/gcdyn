#!/usr/bin/env python
import time
import argparse
import os
import sys
import joblib
import dill
import colored_traceback.always  # noqa: F401
import pickle
import pandas as pd
import random
import csv
import copy
import collections

from gcdyn import utils, encode
from gcdyn.poisson import ConstantResponse

# fmt: off

# ----------------------------------------------------------------------------------------
sum_stat_scaled = {
    "total_branch_length": True
}  # whether to scale summary stats with branch length
smplists = {'train' : ["train", "test"], 'infer' : ['infer']}
default_param_vals = {'yshift' : 0}

# ----------------------------------------------------------------------------------------
def csvfn(args, smpl):
    return "%s/%s.csv" % (args.outdir, smpl)

# ----------------------------------------------------------------------------------------
class LScaler(object):
    # ----------------------------------------------------------------------------------------
    # if either inputs are set, fit/create a new scaler; otherwise you should pass in an existing <scaler>
    # inputs can be either <in_tensors> (encoded trees or fitnesses [latter for per-cell model], which need to be converted to a form that the internal scaler can work on), or <in_vals> which are directly usable by the internal scaler
    def __init__(self, args, var_list, in_tensors=None, in_vals=None, smpl='', scaler=None, dont_scale_params=False):
        self.args = args
        self.var_list = var_list
        self.scaler = scaler
        self.in_tensors = in_tensors
        self.in_vals = in_vals
        self.dont_scale_params = dont_scale_params
        if self.in_vals is not None and len(self.in_vals[0]) != len(self.var_list):
            raise Exception('each column in in_vals should correspond to a variable to scale, but got %d columns and %d vars' % (len(self.in_vals[0]), len(self.var_list)))
        if self.in_tensors is not None or self.in_vals is not None:  # taking input and scaling it
            assert self.in_vals is None or self.in_tensors is None  # only set one of them
            if self.in_tensors is not None:
                self.in_vals = self.extract_tree_vals(self.in_tensors)  # in_vals and out_vals are formatted for scaling (list of rows [one for each node], each row with an entry for each variable)
            self.out_vals = self.scale(self.in_vals, smpl=smpl)
            if self.in_tensors is not None:
                self.out_tensors = self.re_encode_tree_vals(self.out_vals, self.in_tensors)

    # ----------------------------------------------------------------------------------------
    # convert encoded trees to one-variable-per-column format needed by scaler (reverse of re_encode_tree_vals())
    def extract_tree_vals(self, matrix_list):
        rvals = []
        for etree in matrix_list:
            ndicts = encode.decode_matrix('tree', etree)
            rvals += [[n[k] for k in self.var_list] for n in ndicts]
        return rvals

    # ----------------------------------------------------------------------------------------
    # convert values formatted for scaling <scale_vals> (each entry in first list is a list with an entry for each parameter value) back to the original list of matrices format specified by <template_matrices>
    # NOTE template matrix is used for 1) shape or resulting matrix and 2) locations of unfilled entries, i.e. values of <empty_val>
    def re_encode_tree_vals(self, scale_vals, template_matrices, mtype='tree'):
        # ----------------------------------------------------------------------------------------
        def set_entries(tkey, tmtx, nmtx):
            for ntype in ['leaf', 'internal']:
                irow = encode.imtxs[mtype][tkey][ntype]
                for icol in range(len(nmtx[irow])):
                    if encode.is_empty(tmtx[irow][icol]):  # if this entry wasn't filled (i.e. it doesn't correspond to a node)
                        continue
                    nmtx[irow][icol] = scale_vals[iglobals[tkey]][self.var_list.index(tkey)]  # [0] is because we only scale one parameter (fitness) for per-cell prediction (whereas for sigmoid we scaled the three sigmoid parameters)
                    iglobals[tkey] += 1
        # ----------------------------------------------------------------------------------------
        assert self.var_list == ['distance', 'phenotype']  # would need to change things if not
        iglobals = {k : 0 for k in self.var_list}  # index in 1-d list <scale_vals> (don't actually need two of them, they're the same, but this way it's easier to not increment twice)
        new_trees = [copy.deepcopy(tmtx) for tmtx in template_matrices]
        for tkey in self.var_list:
            for tmtx, nmtx in zip(template_matrices, new_trees):
                set_entries(tkey, tmtx, nmtx)
        return new_trees

    # ----------------------------------------------------------------------------------------
    # apply existing scaling to new tensors <in_tensors> or values <in_vals> (see description of difference in __init__ above)
    def apply_scaling(self, in_tensors=None, in_vals=None, smpl='', debug=True):
        assert self.scaler is not None or self.dont_scale_params
        if in_vals is None:
            in_vals = self.extract_tree_vals(in_tensors)
        else:
            assert in_tensors is None
        out_vals = self.scale(in_vals, smpl=smpl)
        if in_tensors is None:
            return out_vals
        else:
            out_tensors = self.re_encode_tree_vals(out_vals, in_tensors)
            return out_tensors

    # ----------------------------------------------------------------------------------------
    def scale(self, invals, inverse=False, smpl='', debug=True):
        """Scale <invals> to mean 0 and variance 1.
        To reverse a scaling, pass in the original scaler and set inverse=True.
        <invals> should be a list of rows, where each row has an entry for each parameter value.
        """
        # ----------------------------------------------------------------------------------------
        def print_debug(pvals_before, pvals_scaled):
            def get_lists(pvs,):  # picks values from rows/columns to get a list of values for each parameter
                return [[plist[ivar] for plist in pvs]]
            def fnstr(pvs, fn):  # apply fn to each list from get_lists(), returns resulting combined str
                return " ".join(("%8.2f" if fn(vl)<1e3 else '%8.1e') % fn(vl) for vl in get_lists(pvs))
            assert self.args.model_type in ['sigmoid', 'per-cell', 'per-bin']
            for ivar, vname in enumerate(self.var_list):
                for dstr, pvs in zip(("before", "after"), (pvals_before, pvals_scaled)):
                    bstr = "   " if dstr != "before" else "      %15s %7s" % (vname, smpl)
                    print("%s %s%s %s%s" % (bstr, fnstr(pvs, sys.modules['numpy'].mean), fnstr(pvs, sys.modules['numpy'].var), fnstr(pvs, min), fnstr(pvs, max), ), end="" if dstr == "before" else "\n")
        # ----------------------------------------------------------------------------------------
        if self.dont_scale_params:
            return copy.copy(invals)
        if inverse:
            assert self.scaler is not None
        if self.scaler is None:  # note: fits each column separately (i.e. each row should contain one value for each parameter/variable)
            self.scaler = sys.modules['sklearn.preprocessing'].StandardScaler().fit(invals)
        sc_pvals = self.scaler.inverse_transform(invals) if inverse else self.scaler.transform(invals)
        if debug:
            if debug:  # and smpl == smplists['train'][0]:
                print("    %sscaling %d variables: %s" % ("reverse " if inverse else "", len(self.var_list), ' '.join(self.var_list)))
                print("                                         before                             after")
                print("                                  mean    var      min    max          mean    var      min    max")
            print_debug(invals, sc_pvals)
        return sc_pvals

# ----------------------------------------------------------------------------------------
def collapse_bundles(args, resps, sstats, gcids):
    # ----------------------------------------------------------------------------------------
    def group_vals(tkey, istart):
        mfcn = min if tkey == "tree" else sys.modules['numpy'].mean  # 'tree' means it's the tree index, in which case we want the index of the first tree in the bundle, otherwise we want the mean (although it should of course be the same for all trees in the bundle)
        valstrs = [sstats[istart + j][tkey] for j in range(args.dl_bundle_size)]
        if '' in valstrs:  # probably real data
            return None
        return mfcn([float(v) for v in valstrs])
    # ----------------------------------------------------------------------------------------
    if resps is not None:
        resps = [
            sys.modules['gcdyn.nn'].ParamNetworkModel._collapse_identical_list(resps[i : i + args.dl_bundle_size])
            for i in range(0, len(resps), args.dl_bundle_size)
        ]
    if sstats is not None:
        sstats = [
            {tkey : group_vals(tkey, i) for tkey in sstats[i]}
            for i in range(0, len(sstats), args.dl_bundle_size)
        ]  # mean of each summary stat over trees in each bundle ('tree' key is an index, so take min/index of first one)
    return resps, sstats, gcids

# ----------------------------------------------------------------------------------------
def write_sigmoid_prediction(args, pred_vals, sstats, gcids, smpl, true_resps=None):
    # ----------------------------------------------------------------------------------------
    def get_empty_df(ptype, plist):
        return {"%s-%s" % (param, ptype): [] for param in plist}
    # ----------------------------------------------------------------------------------------
    dfdata = get_empty_df('predicted', utils.sigmoid_params)
    if true_resps is not None:
        dfdata.update(get_empty_df('truth', true_resps[0]._param_dict)) #{"%s-%s" % (param, 'truth') : [] for param in true_resps[0]._param_dict})
    dfdata['tree-index'] = []
    dfdata['gcids'] = []
    assert true_resps is None or len(pred_vals) == len(true_resps)
    for itr, prlist in enumerate(pred_vals):
        dfdata['tree-index'].append(sstats[itr]['tree'])
        dfdata['gcids'].append(gcids[itr])
        assert len(prlist) == len(args.params_to_predict)
        for ip, param in enumerate(args.params_to_predict):
            dfdata["%s-predicted" % param].append(prlist[ip])
        for xpm in [p for p in utils.sigmoid_params if p not in args.params_to_predict]:
            dfdata["%s-predicted" % xpm].append(default_param_vals[xpm])
        if true_resps is not None:
            for param in true_resps[itr]._param_dict:
                dfdata["%s-truth" % param].append(get_pval(param, true_resps[itr], sstats[itr]))
    df = pd.DataFrame(dfdata)
    print("  writing %s results to %s" % (smpl, args.outdir))
    df.to_csv(csvfn(args, smpl))
    return df

# ----------------------------------------------------------------------------------------
def write_per_cell_prediction(args, pred_fitnesses, enc_trees, true_fitnesses=None, true_resps=None, sstats=None, smpl=None):
    assert true_fitnesses is None or len(pred_fitnesses) == len(true_fitnesses)
    dfdata = {  # make empty df
        "fitness-%s"%ptype : []
        for ptype in (["predicted"] if true_fitnesses is None else ["truth", "predicted"])
    }
    dfdata['tree-index'] = []
    dfdata['phenotype'] = []
    if true_fitnesses is not None:
        for param in utils.sigmoid_params:
            dfdata['%s-truth'%param] = []
    for itr, efit in enumerate(pred_fitnesses):
        pred_ndicts = encode.decode_matrices(enc_trees[itr], efit)
        if true_fitnesses is not None:
            true_ndicts = encode.decode_matrices(enc_trees[itr], true_fitnesses[itr])
            assert len(pred_ndicts) == len(true_ndicts)
        for icell, pdict in enumerate(pred_ndicts):
            pdict['fitness-predicted'] = pdict['fitness']
            del pdict['fitness']
            if true_fitnesses is not None:
                pdict['fitness-truth'] = true_ndicts[icell]['fitness']
                for ip, param in enumerate(utils.sigmoid_params):  # writes true parameter values for each cell, which kind of sucks, but they're only nonzero for the first cell in each tree
                    dfdata["%s-truth" % param].append(get_pval(param, true_resps[itr], sstats[itr]) if icell==0 else 0)
        for ndict in pred_ndicts:
            dfdata['tree-index'].append(sstats[itr]['tree'])  # NOTE tree-index isn't necessarily equal to <itr>
            for tk in [k for k in ndict if k in dfdata]:
                dfdata[tk].append(ndict[tk])
    df = pd.DataFrame(dfdata)
    print("  writing %s results to %s" % (smpl, args.outdir))
    df.to_csv(csvfn(args, smpl))
    return df

# ----------------------------------------------------------------------------------------
def write_per_bin_prediction(args, pred_fitness_bins, enc_trees, true_fitness_bins=None, true_resps=None, sstats=None, gcids=None, smpl=None):
    assert true_fitness_bins is None or len(pred_fitness_bins) == len(true_fitness_bins)
    dfdata = {  # make empty df
        "fitness-bins-%s-ival-%d"%(ptype, ival) : []
        for ptype in (["predicted"] if true_fitness_bins is None else ["truth", "predicted"])
        for ival in range(len(utils.zoom_affy_bins)-1)
    }
    dfdata['tree-index'] = []
    dfdata['gcids'] = []
    if true_fitness_bins is not None:
        for param in true_resps[0]._param_dict:
            dfdata['%s-truth'%param] = []
    for itr, pfbins in enumerate(pred_fitness_bins):
        dfdata['tree-index'].append(sstats[itr]['tree'])  # NOTE tree-index isn't necessarily equal to <itr>
        dfdata['gcids'].append(gcids[itr])
        for ival, bval in enumerate(pfbins):
            dfdata['fitness-bins-predicted-ival-%d'%ival].append(bval)  # it kind of sucks to expand these out to a column each, but the other options are a) make a new row for each one or b) more complicated csv formatting, both of which are worse
            if true_fitness_bins is not None:
                assert len(pfbins) == len(true_fitness_bins[itr])
                dfdata['fitness-bins-truth-ival-%d'%ival].append(true_fitness_bins[itr][ival])
        if true_fitness_bins is not None:
            for param in true_resps[itr]._param_dict:  # writes true parameter values for each cell, which kind of sucks, but they're only nonzero for the first cell in each tree
                dfdata["%s-truth" % param].append(get_pval(param, true_resps[itr], sstats[itr]))
    df = pd.DataFrame(dfdata)
    print("  writing %s results to %s" % (smpl, args.outdir))
    df.to_csv(csvfn(args, smpl))
    return df

# ----------------------------------------------------------------------------------------
def get_prediction(args, model, spld, lscalers, smpl=None):
    true_fitnesses, true_fitness_bins, true_resps, sstats = None, None, None, None
    sstats = spld['sstats']
    if args.is_simu:
        in_vals = [[float(d[k]) for k in utils.non_sigmoid_input_params] for d in spld['sstats']]
    else:
        in_vals = [[args.carry_cap_values, args.init_population_values, args.death_values] for _ in spld['trees']]
    carry_caps, init_pops, deaths = zip(*lscalers['per-tree'].apply_scaling(in_vals=in_vals))
    if not args.is_simu:
        print('    carry caps: %d --> %s' % (args.carry_cap_values, carry_caps[:5]))
        print('    init pops: %d --> %s' % (args.init_population_values, init_pops[:5]))
        print('    deaths: %d --> %s' % (args.death_values, deaths[:5]))
    gcids = spld['gcids']
    if args.model_type == 'sigmoid':
        const_pred_resps = model.predict(lscalers['per-node'].apply_scaling(in_tensors=spld['trees'], smpl=smpl), carry_caps, init_pops, deaths)  # note that this returns constant response fcns that are just holders for the predicted values (i.e. don't directly relate to true/input response fcns)
        pred_vals = [[float(rsp.value) for rsp in rlist] for rlist in const_pred_resps]
        pred_vals = lscalers['output'].scale(pred_vals, inverse=True, smpl=smpl)
        if args.is_simu:
            true_resps = spld["birth-responses"]
        if args.dl_bundle_size is not None:
            raise Exception('need to handle gcids (also need better default gcids in previous lines)')
            true_resps, sstats, gcids = collapse_bundles(args, true_resps, sstats, gcids)
        if args.is_simu:
            assert len(pred_vals) == len(true_resps)
        df = write_sigmoid_prediction(args, pred_vals, sstats, gcids, smpl, true_resps=true_resps)
    elif args.model_type == 'per-cell':
        assert args.dl_bundle_size is None
        pred_fitnesses = model.predict(lscalers['per-node'].apply_scaling(in_tensors=spld['trees'], smpl=smpl)).numpy()
        for pfit, etree in zip(pred_fitnesses, spld['trees']):
            encode.reset_fill_entries(pfit, etree)
        if args.is_simu:
            true_fitnesses, true_resps = [spld[tk] for tk in ["fitnesses", "birth-responses"]]
        df = write_per_cell_prediction(args, pred_fitnesses, spld["trees"], true_fitnesses=true_fitnesses, true_resps=true_resps, sstats=sstats, smpl=smpl)
    elif args.model_type == 'per-bin':
        assert args.dl_bundle_size is None
        pred_fitness_bins = model.predict(lscalers['per-node'].apply_scaling(in_tensors=spld['trees'], smpl=smpl), carry_caps, init_pops, deaths).numpy()
        pred_fitness_bins = encode.wrap_fitness_bins(lscalers['output'].scale(encode.unwrap_fitness_bins(pred_fitness_bins), inverse=True, smpl=smpl), len(pred_fitness_bins[0]))
        if args.is_simu:
            true_fitness_bins, true_resps = [spld[tk] for tk in ["fitness-bins", "birth-responses"]]
        df = write_per_bin_prediction(args, pred_fitness_bins, spld["trees"], true_fitness_bins=true_fitness_bins, true_resps=true_resps, sstats=sstats, smpl=smpl, gcids=gcids)
    else:
        assert False
    return df

# ----------------------------------------------------------------------------------------
def plot_existing_results(args):
    prdfs = {}
    for smpl in smplists[args.action]:
        prdfs[smpl] = pd.read_csv(csvfn(args, smpl))
    seqmeta = read_csv('%s/meta.csv'%args.indir)
    sstats = read_csv('%s/summary-stats.csv'%args.indir)
    utils.make_dl_plots(
        args.model_type,
        prdfs,
        seqmeta,
        sstats,
        args.params_to_predict,
        args.outdir + "/plots",
        is_simu=args.is_simu,
        validation_split=args.validation_split,
        trivial_encoding=args.use_trivial_encoding,
        force_many_plot=args.force_many_plot,
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
def write_traintest_samples(args, smpldict):
    for smpl in smplists['train']:
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
def read_csv(mfn):
    metafos = []
    with open(mfn) as lmfile:
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
    rfn, tfn, ffn, sfn, gfn = [
        "%s/%s" % (args.indir, s)
        for s in ["responses.pkl", "encoded-trees.npy", "encoded-fitnesses.npy", "summary-stats.csv", 'gcids.csv']
    ]
    rstr = ''
    if args.is_simu:
        with open(rfn, "rb") as rfile:
            pklfo = pickle.load(rfile)
        rstr = ' (%d responses from %s)' % (len(pklfo), rfn)
        for tk in ["birth", "death"]:
            samples[tk + "-responses"] = [tfo[tk] for tfo in pklfo]
    samples["trees"] = encode.read_trees(tfn)
    samples['trees'] = encode.pad_trees(samples['trees'], 'tree', args.min_n_max_leaves)
    if args.is_simu:
        samples["fitnesses"] = encode.read_trees(ffn)
        samples["fitness-bins"] = encode.read_trees(ffn.replace('fitnesses', 'fitness-bins'))
    samples["sstats"] = read_csv(sfn)
    if os.path.exists(gfn):  # name/label for each tree/gc (only for data)
        with open(gfn) as gfile:
            samples['gcids'] = [l['gcid'] for l in csv.DictReader(gfile)]
    else:
        samples['gcids'] = ['' for _ in samples['sstats']]
    assert len(samples['gcids']) == len(samples['trees'])
    print("    read %d trees from %s%s" % (len(samples["trees"]), tfn, rstr))
    if args.is_simu:
        print("      first response pair:\n        birth: %s\n        death: %s" % (samples["birth-responses"][0], samples["death-responses"][0]))
        if args.dl_bundle_size is not None:
            check_bundles(samples)
    if args.dl_bundle_size is not None and len(samples["trees"]) % args.dl_bundle_size != 0:
        if args.discard_extra_trees:
            n_remain = len(samples["trees"]) % args.dl_bundle_size
            print('  --discard-extra-trees: discarding %d trees from end of input since total N trees %d isn\'t evenly divisible by bundle size %d' % (n_remain, len(samples['trees']), args.dl_bundle_size))
            for tk in samples:
                samples[tk] = samples[tk][: len(samples[tk]) - n_remain]
        else:
            raise Exception('N trees %d not divisible by bundle size %d' % (len(samples["trees"]), args.dl_bundle_size))
    if args.n_max_trees is not None:
        for tk in samples:
            assert args.n_max_trees < len(samples[tk])
            n_before = len(samples[tk])
            samples[tk] = samples[tk][:args.n_max_trees]
            assert len(samples[tk]) == args.n_max_trees
        print('    --n-max-trees: only using first %d / %d trees' % (len(samples[tk]), n_before))
    if args.is_simu and args.resample_param is not None:
        n_max = 100  # max denominator in fraction to skip (just to make math easier below)
        def gpv(i): return getattr(samples['birth-responses'][i], args.resample_param)
        def gpvs(): return [gpv(i) for i in range(len(samples['trees']))]
        def prdbg(tstr, skpairs, skhist, before_hist=None):
            print('    %s: %d total' % (tstr, len(gpvs())))
            print('       low edge: %s' % ' '.join('%7.2f'%skhist.low_edges[i] for i in skhist.ibiniter(False)))
            print('      skip frac:    %s' % ' '.join('%7.2f'%f for _, f in skpairs))
            print('         N vals:    %s' % ' '.join('%7d'%skhist.bin_contents[i] for i in skhist.ibiniter(False)))
            if before_hist is not None:
                fdvals = [(before_hist.bin_contents[i] - skhist.bin_contents[i])/before_hist.bin_contents[i] if before_hist.bin_contents[i]!=0 else 0 for i in skhist.ibiniter(False)]
                print('         N vals:    %s' % ' '.join('%7.2f'%v for v in fdvals))
        skpairs = []
        for skpstr in args.resample_cfg.split(':'):
            skip_val, skfracstr = skpstr.split(',')
            skip_val = float(skip_val)
            if '/' in skfracstr:
                num, denom = [int(v) for v in skfracstr.split('/')]
                assert denom < n_max  # just to make math easier below
                skip_frac = float(num / denom)
            else:
                skip_frac = float(skfracstr)
            skpairs.append([skip_val, skip_frac])
        xmin = min(gpvs())-0.1
        skhist = utils.Hist(n_bins=len(skpairs), xmin=xmin, xmax=skpairs[-1][0], xbins=[xmin] + [v for v, _ in skpairs], value_list=gpvs())
        prdbg('before', skpairs, skhist)
        
        new_samples = {k : [] for k in samples}
        n_skipped = 0
        for ismpl in range(len(samples['trees'])):
            ibin = skhist.find_bin(gpv(ismpl))
            skip_val, skip_frac = skpairs[ibin-1]
            if random.randint(0, n_max) < skip_frac * n_max:
                n_skipped += 1
                continue
            for tk in samples:
                new_samples[tk].append(samples[tk][ismpl])
        samples = new_samples
        fin_hist = utils.Hist(n_bins=len(skpairs), xmin=xmin, xmax=skpairs[-1][0], xbins=[xmin] + [v for v, _ in skpairs], value_list=gpvs())
        prdbg('after', skpairs, fin_hist, before_hist=skhist)
        print('  --resample-param %s: skipped %d / %d trees using (kept %d)' % (args.resample_param, n_skipped, n_skipped + len(samples[tk]), len(samples[tk])))

    return samples

# ----------------------------------------------------------------------------------------
def predict_and_plot(args, model, smpldict, smpls, lscalers=None):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    prdfs = {}
    for smpl in smpls:
        prdfs[smpl] = get_prediction(args, model, smpldict[smpl], lscalers, smpl=smpl)
    seqmeta = read_csv('%s/meta.csv'%args.indir)
    sstats = read_csv('%s/summary-stats.csv'%args.indir)
    utils.make_dl_plots(  # note that response bundles are collapsed (i.e. we only plot the first pair of each bundle) but seqmeta isn't, so we just plot the affinities from the first tree in the bundle (it would probably be better to combine all of the affinities from the bundle)
        args.model_type,
        prdfs,
        seqmeta,
        sstats,
        args.params_to_predict,
        args.outdir + "/plots",
        is_simu=args.is_simu,
        validation_split=0 if smpl=='infer' else args.validation_split,
        trivial_encoding=args.use_trivial_encoding,
        force_many_plot=args.force_many_plot,
    )

# ----------------------------------------------------------------------------------------
def train_and_test(args, start_time):
    # ----------------------------------------------------------------------------------------
    def train_sigmoid(smpldict, lscalers, max_leaf_count):
        responses = [[ConstantResponse(p) for p in plist] for plist in lscalers['train']['output'].out_vals]  # order corresponds to args.params_to_predict (constant response is just a container for one value, and note we don't bother to set the name)
        model = sys.modules['gcdyn.nn'].ParamNetworkModel(responses[0], bundle_size=args.dl_bundle_size, custom_loop=args.custom_loop, params_to_predict=args.params_to_predict)
        model.build_model(max_leaf_count, dropout_rate=args.dropout_rate, learning_rate=args.learning_rate, ema_momentum=args.ema_momentum, prebundle_layer_cfg=args.prebundle_layer_cfg, loss_fcn=args.loss_fcn)
        carry_caps, init_pops, deaths = zip(*lscalers['train']['per-tree'].out_vals)
        model.fit(lscalers['train']['per-node'].out_tensors, responses, carry_caps, init_pops, deaths, epochs=args.epochs, batch_size=args.batch_size, validation_split=args.validation_split)
        return model
    # ----------------------------------------------------------------------------------------
    def train_per_cell(smpldict, lscalers, max_leaf_count):
        model = sys.modules['gcdyn.nn'].PerCellNetworkModel()
        model.build_model(max_leaf_count, dropout_rate=args.dropout_rate, learning_rate=args.learning_rate, ema_momentum=args.ema_momentum, loss_fcn=args.loss_fcn)
        model.fit(lscalers['train'].out_tensors, smpldict['train']['fitnesses'], epochs=args.epochs, batch_size=args.batch_size, validation_split=args.validation_split)
        return model
    # ----------------------------------------------------------------------------------------
    def train_per_bin(smpldict, lscalers, max_leaf_count):
        model = sys.modules['gcdyn.nn'].PerBinNetworkModel()
        model.build_model(len(utils.zoom_affy_bins) - 1, max_leaf_count, dropout_rate=args.dropout_rate, learning_rate=args.learning_rate, ema_momentum=args.ema_momentum, prebundle_layer_cfg=args.prebundle_layer_cfg, loss_fcn=args.loss_fcn)
        carry_caps, init_pops, deaths = zip(*lscalers['train']['per-tree'].out_vals)
        scaled_fitness_bins = encode.wrap_fitness_bins(lscalers['train']['output'].out_vals, len(smpldict['train']['fitness-bins'][0]))
        model.fit(lscalers['train']['per-node'].out_tensors, scaled_fitness_bins, carry_caps, init_pops, deaths, epochs=args.epochs, batch_size=args.batch_size, validation_split=args.validation_split)
        return model
    # ----------------------------------------------------------------------------------------
    samples = read_tree_files(args)

    # separate train/test samples
    idxs = get_traintest_indices(args, samples)
    smpldict = {}  # separate train/test trees, responses, etc by index
    for smpl in smplists['train']:
        smpldict[smpl] = {
            key: [val[i] for i in idxs[smpl]] for key, val in samples.items()
        }
    print("      N trees: %s" % "   ".join("%s %d" % (s, len(smpldict[s]["trees"])) for s in smplists['train']))

    write_traintest_samples(args, smpldict)

    leaf_counts = set([len(t[0]) for t in smpldict["train"]["trees"]])  # length of first row in encoded tree (i guess really it'd be better to also include test trees in this, but in practice it probably doesn't matter)
    if len(leaf_counts) != 1:
        raise Exception("encoded trees have different lengths: %s" % " ".join(str(c) for c in leaf_counts))
    max_leaf_count = max(list(leaf_counts)[0], args.min_n_max_leaves)

    # handle various scaling/re-encoding stuff
    lscalers = {}
    for smpl in smplists['train']:
        lscalers[smpl] = {'per-node' : LScaler(args, ['distance', 'phenotype'], in_tensors=smpldict[smpl]['trees'], smpl=smpl, dont_scale_params=args.dont_scale_input_params)}
        lscalers[smpl]['per-tree'] = LScaler(args, utils.non_sigmoid_input_params, in_vals=[[float(d[k]) for k in utils.non_sigmoid_input_params] for d in smpldict[smpl]['sstats']], smpl=smpl, dont_scale_params=args.dont_scale_input_params)
        if args.model_type == 'sigmoid':
            lscalers[smpl]['output'] = LScaler(args, args.params_to_predict, in_vals=[[getattr(r, p) for p in args.params_to_predict] for r in smpldict[smpl]['birth-responses']], smpl=smpl, dont_scale_params=not args.scale_output_params)
        else:
            lscalers[smpl]['output'] = LScaler(args, ['fitness-bins'], in_vals=encode.unwrap_fitness_bins(smpldict[smpl]['fitness-bins']), smpl=smpl, dont_scale_params=not args.scale_output_params)
    for tstr in ['per-node', 'per-tree', 'output']:
        joblib.dump(lscalers['train'][tstr].scaler, encode.output_fn(args.outdir, '%s-train-scaler'%tstr, None))

    # silly encodings for testing that essentially train on the output values
    if args.use_trivial_encoding:
        for smpl in smplists['train']:
            if args.model_type == 'per-cell':
                predict_vals = smpldict[smpl]['fitnesses']
            else:
                predict_vals = [[getattr(rsp, p) for p in args.params_to_predict] for rsp in smpldict['train']['birth-responses']]  # order corresponds to args.params_to_predict (constant response is just a container for one value, and note we don't bother to set the name)
            encode.trivialize_encodings(smpldict[smpl]["trees"], args.model_type, predict_vals, noise=False, n_debug=3)

    if args.model_type == 'sigmoid':
        model = train_sigmoid(smpldict, lscalers, max_leaf_count)
    elif args.model_type == 'per-cell':
        model = train_per_cell(smpldict, lscalers, max_leaf_count)
    elif args.model_type == 'per-bin':
        model = train_per_bin(smpldict, lscalers, max_leaf_count)
    else:
        assert False
    model.network.save(encode.output_fn(args.outdir, 'model', None))

    predict_and_plot(args, model, smpldict, ['train', 'test'], lscalers=lscalers["train"])
    print("    total dl inference time: %.1f sec" % (time.time() - start_time))

# ----------------------------------------------------------------------------------------
def read_model_files(args, samples):
    scfns = {tstr : encode.output_fn(args.model_dir, '%s-train-scaler'%tstr, None) for tstr in ['per-node', 'per-tree', 'output']}
    print('    reading training scalers from %s' % ' '.join(scfns.values()))
    lscalers = {'per-node' : LScaler(args, ['distance', 'phenotype'], scaler=joblib.load(scfns['per-node']), dont_scale_params=args.dont_scale_input_params)}
    lscalers['per-tree'] = LScaler(args, utils.non_sigmoid_input_params, scaler=joblib.load(scfns['per-tree']), dont_scale_params=args.dont_scale_input_params)
    assert args.model_type in ['sigmoid', 'per-bin']
    lscalers['output'] = LScaler(args, args.params_to_predict if args.model_type=='sigmoid' else ['fitness-bins'], scaler=joblib.load(scfns['output']), dont_scale_params=not args.scale_output_params)
    if args.model_type == 'sigmoid':
        model = sys.modules['gcdyn.nn'].ParamNetworkModel([ConstantResponse(0) for _ in args.params_to_predict], bundle_size=args.dl_bundle_size, custom_loop=args.custom_loop, params_to_predict=args.params_to_predict)
    elif args.model_type == 'per-cell':
        model = sys.modules['gcdyn.nn'].PerCellNetworkModel()
    elif args.model_type == 'per-bin':
        model = sys.modules['gcdyn.nn'].PerBinNetworkModel()
    else:
        assert False
    model.load(encode.output_fn(args.model_dir, 'model', None))
    return lscalers, model

# ----------------------------------------------------------------------------------------
def infer(args, start_time):
    smpldict = {'infer' : read_tree_files(args)}
    lscalers, model = read_model_files(args, smpldict['infer'])
    predict_and_plot(args, model, smpldict, ['infer'], lscalers=lscalers)
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
    parser.add_argument("--model-type", choices=['sigmoid', 'per-cell', 'per-bin'], default='sigmoid', help='type of neural network model, sigmoid: infer 3 params of sigmoid fcn, per-cell: infer fitness of each individual cell, per-bin: infer response function shape in bins of affinity')
    parser.add_argument("--loss-fcn", choices=['mse', 'curve', 'per-cell-masked'], default='curve')
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dl-bundle-size", type=int, help='\'dl-\' is to differentiate from \'simu-\' bundle size when calling this from cf-gcdyn.py')
    parser.add_argument("--discard-extra-trees", action="store_true", help='By default, the number of trees during inference must be evenly divisible by --dl-bundle-size. If this is set, however, any extras are discarded to allow inference.')
    parser.add_argument("--dropout-rate", type=float, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--prebundle-layer-cfg", default='default') #, choices=['default', 'small', 'big', 'huge'])
    parser.add_argument("--train-frac", type=float, default=0.8, help="train on this fraction of the trees")
    parser.add_argument("--validation-split", type=float, default=0.1, help="fraction of training sample to tell keras to hold out for validation during training")
    parser.add_argument("--params-to-predict", default=utils.sigmoid_params, nargs="+", choices=["xscale", "xshift", "yscale", "yshift"] + [k for k in sum_stat_scaled])
    parser.add_argument("--resample-param", default='yscale', choices=utils.sigmoid_params, help='parameter to use for resampling simulation (i.e. use the value of this parameter to decide which GCs to keep/discard for training). See also --resample-cfg.')
    parser.add_argument("--resample-cfg", default='5,0:10,1/2:15,1/2:20,1/2:30,1/2:40,1/2', help='configure how to resample --resample-param: colon-separated list of comma-separated pairs, with first element of each pair the upper edge of a bin, and the second element the fraction of entries to discard in that bin.')
    parser.add_argument("--test", action="store_true", help="sets things to be super fast, so not useful for real inference, but just to check if things are running properly")
    parser.add_argument("--carry-cap-values", type=int, help="input parameter value for data inference (single valued, it\'s just plural to avoid making a new arg in cf-gcdyn.py)")
    parser.add_argument("--init-population-values", type=int, help="input parameter value for data inference (single valued, it\'s just plural to avoid making a new arg in cf-gcdyn.py)")
    parser.add_argument("--death-values", type=float, help="input parameter value for data inference (single valued, it\'s just plural to avoid making a new arg in cf-gcdyn.py)")
    parser.add_argument("--is-simu", action="store_true", help="set to this if running on simulation")
    parser.add_argument("--random-seed", default=0, type=int, help="random seed")
    parser.add_argument("--n-max-trees", type=int, help="if set, after reading this many trees from input, discard any extras")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-trivial-encoding", action="store_true")
    parser.add_argument("--dont-scale-input-params", action="store_true")
    parser.add_argument("--scale-output-params", action="store_true")
    parser.add_argument("--min-n-max-leaves", default=200, help='pad all encoded tree matrices to at least this width')
    parser.add_argument("--custom-loop", action="store_true")
    parser.add_argument("--force-many-plot", action="store_true", help='make plot with response functions for all GCs on top of each other, even if this isn\'t simulation and/or they don\'t all have the same parameters.')
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
    if args.use_trivial_encoding and not args.dont_scale_input_params:
        print('  %s --use-trivial-encoding: turning on --dont-scale-params since parameter scaling needs fixing to work with trivial encoding' % utils.wrnstr())
        args.dont_scale_input_params = True
    if args.model_dir is not None and args.action == 'train':
        raise Exception('doesn\'t make sense to set --model-dir when training')
    if args.model_type == 'per-bin' and args.loss_fcn == 'curve':
        print('    note: setting --loss-fcn to \'mean abs relative error\' for per-bin model (this var doesn\'t have any effect atm')
        args.loss_fcn = 'mean abs relative error'
    if args.carry_cap_values is not None or args.init_population_values is not None or args.death_values is not None:
        if args.is_simu:
            raise Exception('can only set carry cap, init population, or death for data (otherwise they come from the simulation files)')
    if [p for p in utils.sigmoid_params if p in args.params_to_predict] != args.params_to_predict:
        raise Exception('--params-to-predict (%s) must be in same order as utils.sigmoid_params (%s)' % (' '.join(args.params_to_predict), ' '.join(utils.sigmoid_params)))
    outputs_exist = all(os.path.exists(csvfn(args, s)) for s in smplists[args.action]) and not args.overwrite

    if not outputs_exist:
        # these imports are super slow, so don't want to wait for them to get help message or plot existing csvs
        import tensorflow as tf
        from gcdyn.nn import ParamNetworkModel, PerCellNetworkModel, PerBinNetworkModel
        from sklearn import preprocessing
    import numpy  # has to be imported *after* tf
    random.seed(args.random_seed)
    numpy.random.seed(args.random_seed)

    if outputs_exist:
        print("    csv files already exist, so just replotting (override with --overwrite): %s" % ' '.join(csvfn(args, s) for s in smplists[args.action]))
        plot_existing_results(args)
        sys.exit(0)

    tf.keras.utils.set_random_seed(args.random_seed)
    if args.action == 'train':
        train_and_test(args, start_time)
    elif args.action == 'infer':
        infer(args, start_time)
    else:
        assert False
# fmt: on
