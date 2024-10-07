import numpy as np
import argparse
import os
import sys
import csv
import colored_traceback.always  # noqa: F401
import pickle
import time
import copy
import random
import subprocess
import dill
import math
import glob
import json

# fmt: off
# TODO rm this path stuff and pip install to go back to using installed bdms-sim (rather than cloned github copy)
#   - pip install bdms-sim
#   - pip uninstall bdms-sim
bdms_dir = os.path.dirname(os.path.realpath(__file__)).replace('/gcdyn/scripts', '/bdms')
sys.path.insert(1, bdms_dir)
from gcdyn import gpmap, mutators, poisson, utils, encode
from bdms import tree as bdmstree
from bdms.poisson import ConstantProcess
from experiments import replay

# partis_dir = os.path.dirname(os.path.realpath(__file__)).replace('/projects/gcdyn/scripts', '')
# sys.path.insert(1, partis_dir) # + '/python')

# ----------------------------------------------------------------------------------------
def outfn(args, ftype, itrial=None, subd=None):
    odir = args.outdir
    if subd is not None:
        odir += '/%s' % subd
    return encode.output_fn(odir, ftype, itrial)

# ----------------------------------------------------------------------------------------
def print_final_response_vals(tree, birth_resp, death_resp, final_time):
    print("                           x         birth           death")
    print("      time   N seqs.   min   max    min  max       min     max")
    xvals, bvals, dvals = [], [], []
    for tval in range(final_time + 1):  # kind of weird/arbitrary to take integer values
        tstatev = sorted(tree.slice(tval))
        tbv, tdv = [[r.λ_homogeneous(tst) for tst in tstatev] for r in [birth_resp, death_resp]]
        txv = [tst.x for tst in tstatev]
        xvals += txv
        bvals += tbv
        dvals += tdv
        print("      %3d   %4d     %5.2f %5.2f  %5.2f %5.2f   %6.3f %6.3f" % (tval, len(txv), min(txv), max(txv), min(tbv), max(tbv), min(tdv), max(tdv)))  # fmt: skip
    print("             mean      min       max")  # fmt: skip
    print("       x   %6.2f    %6.2f    %6.2f" % (np.mean(xvals), min(xvals), max(xvals)))  # fmt: skip
    print("     birth %6.2f    %6.2f    %6.2f" % (np.mean(bvals), min(bvals), max(bvals)))  # fmt: skip
    print("     death %7.3f   %7.3f   %7.3f" % (np.mean(dvals), min(dvals), max(dvals)))  # fmt: skip


# ----------------------------------------------------------------------------------------
def relabel_nodes(args, tree, itrial, only_internal=False, seqfos=None):
    if seqfos is not None:
        seqdict = {s['name'] : s for s in seqfos}
    for node in [tree] + list(tree.iter_descendants()):
        if only_internal and node.is_leaf():
            continue
        old_name = node.name
        if node.is_root():
            node.name = "naive"
        elif args.label_leaf_internal_nodes:  # don't want naive node to also have 'mrca'
            node.name = '%s-%s' % ('leaf' if node.is_leaf() else 'mrca', node.name)
        node.name = "%d-%s" % (itrial, node.name)
        if seqfos is not None and old_name in seqdict:
            seqdict[old_name]['name'] = node.name

class NodeState(object):  # kinda essentially used as a hashable node with no name (representing all nodes with common state, i.e. atm common x + sequence)
    def __init__(self, **kwargs):
        self.state_attrs = ('x', 'sequence', 'chain_2_start_idx')
        for attr in kwargs:
            assert attr in self.state_attrs
            setattr(self, attr, kwargs[attr])
    def __str__(self):
        return '   x: %.2f  idx: %3d seq: %s' % (self.x, self.chain_2_start_idx, self.sequence)
    def __lt__(self, other):
        return self.x < other.x

# ----------------------------------------------------------------------------------------
def generate_sequences_and_tree(
    args,
    params,
    birth_resp,
    death_resp,
    mutation_resp,
    mutator,
    gp_map,
    itrial,
    seed=0,
):
    err_strs, success = {}, False
    for itry in range(args.n_max_tries):
        try:
            tree_start = time.time()
            nstate = NodeState(x=gp_map(replay.NAIVE_SEQUENCE), sequence=replay.NAIVE_SEQUENCE, chain_2_start_idx=replay.CHAIN_2_START_IDX)
            tree = bdmstree.TreeNode(state=nstate)
            tree.evolve(
                params['time_to_sampling'],
                birth_process=birth_resp,
                death_process=death_resp,
                mutation_process=mutation_resp,
                mutator=mutator,
                min_survivors=args.min_survivors,
                birth_mutation_prob=0,
                capacity=params["carry_cap"],
                capacity_method=args.capacity_method,
                init_population=args.init_population,
                seed=seed,
                verbose=args.debug > 1,
            )
            live_leaves = [l for l in tree if l.event == tree._SURVIVAL_EVENT]
            print(
                "    finished tree with %d live tips (%d total tips) at time %.1f%s (%.1f sec)"
                % (len(live_leaves), len(tree), np.mean([l.t for l in tree.iter_leaves()]), '' if itry==0 else '  try %d' % (itry + 1),  time.time() - tree_start)
            )
            if args.debug:
                print_final_response_vals(tree, birth_resp, death_resp, params['time_to_sampling'])
            success = True
            break
        except bdmstree.TreeError as terr:
            estr = terr.value
            if "min_survivors" in estr:
                estr = "min survivors too small (less than %d)" % args.min_survivors
            # elif terr.value.find('maximum number of leaves') == 0:
            #     max_fails += 1
            if estr not in err_strs:
                err_strs[estr] = 0
            err_strs[estr] += 1
            print(
                "%s%s" % ("failures: " if sum(err_strs.values()) == 1 else "", "."),
                end="",
                flush=True,
            )
            continue
    print()
    for estr in sorted([k for k, v in err_strs.items() if v > 0]):
        print("      %s %d failures with message '%s'" % (utils.color("yellow", "warning"), err_strs[estr], estr))
    if not success:
        print("    %s exceeded maximum number of tries %d so giving up" % (utils.color("yellow", "warning"), args.n_max_tries))
        return None, None

    fnlist = None
    if args.make_plots:
        fnlist = utils.plot_n_vs_time(args.outdir + "/plots/tree-slices", tree, params['time_to_sampling'], itrial)

    if args.debug > 1:
        print('    tree before sampling:')
        utils.print_dtree(tree)
    n_to_sample = params["n_seqs"]
    if len(live_leaves) < n_to_sample:
        print("  %s --n-seqs set to %d but tree only has %d live tips, so just sampling all of them" % (utils.color("yellow", "warning"), n_to_sample, len(live_leaves)))
        n_to_sample = len(live_leaves)
    tree.sample_survivors(n=n_to_sample, seed=seed)
# TODO wait don't we not need to prune any more?
    tree.prune_unsampled()
    tree.remove_mutation_events()
    tree.check_binarity()
    relabel_nodes(args, tree, itrial)
    if args.debug > 1:
        print('    tree after sampling, pruning, relabeling, etc:')
        utils.print_dtree(tree)

    # check that node times and branch lengths are consistent
    for node in tree.iter_descendants():
        assert np.isclose(node.t - node.up.t, node.dist)

    set_mut_stats(tree, debug=args.debug)

    return fnlist, tree


# ----------------------------------------------------------------------------------------
def set_mut_stats(tree, debug=False):
    tree.total_mutations, tree.total_aa_muts = 0, 0
    naive_seq_aa = utils.ltranslate(replay.NAIVE_SEQUENCE)
    for node in tree.iter_descendants(strategy="preorder"):
        node.total_mutations = node.n_mutations + node.up.total_mutations
        node.total_aa_muts = utils.hamming_distance(naive_seq_aa, utils.ltranslate(node.state.sequence), amino_acid=True)

    if debug:
        for tattr, tname in [('total_mutations', 'muts'), ('t', 'times')]:
            int_vals = sorted([getattr(n, tattr) for n in [tree] + list(tree.iter_descendants()) if not n.is_leaf()])
            leaf_vals = sorted([getattr(l, tattr) for l in tree.iter_leaves()])
            def tfn(v): return ('%.1f' if tattr=='t' else '%d') % v
            print('      internal %s (mean %.1f): %s' % (tname, np.mean(int_vals), ' '.join(tfn(n) for n in int_vals)))
            print('          leaf %s (mean %.1f): %s' % (tname, np.mean(leaf_vals), ' '.join(tfn(n) for n in leaf_vals)))

# ----------------------------------------------------------------------------------------
def scan_response(
    birth_resp, death_resp, xmin=-5, xmax=2, nsteps=10
):  # print output values of response function
    dx = (xmax - xmin) / nsteps
    xvals = list(np.arange(xmin, 0, dx)) + list(np.arange(0, xmax + dx, dx))
    rvals = [birth_resp.λ_homogeneous(NodeState(x=x)) for x in xvals]
    xstr = "   ".join("%7.2f" % x for x in xvals)
    rstr = "   ".join("%7.2f" % r for r in rvals)
    print("    x:", xstr)
    print("    r:", rstr)


# ----------------------------------------------------------------------------------------
def print_resp(bresp, dresp):
    print("        response    f(x=0)    function")
    for rname, rfcn in zip(["birth", "death"], [bresp, dresp]):
        print("          %s   %7.3f      %s" % (rname, rfcn.λ_homogeneous(NodeState(x=0)), rfcn))


# ----------------------------------------------------------------------------------------
def choose_val(args, pname, extra_bounds=None, dbgstrs=None):
    minmax, vals = [getattr(args, pname + "_" + str) for str in ["range", "values"]]
    if minmax is not None:  # range with two values for continuous
        minv, maxv = minmax
        if extra_bounds is not None:
            # use the more restrictive (larger lo, smaller hi) values
            minv = max(minv, extra_bounds[0])
            maxv = min(maxv, extra_bounds[1])
        print("        choosing %s within [%.2f, %.2f]%s" % (pname, minv, maxv, '' if (dbgstrs is None or len(dbgstrs)==0) else ' (%s)'%', '.join(dbgstrs)))
        if minv > maxv:
            raise Exception('arrived at nonsense range for %s: [%.2f, %.2f]' % (pname, minv, maxv))
        if pname in ["time_to_sampling", 'carry-cap']:
            return np.random.choice(range(minv, maxv + 1))  # integers (note that this is inclusive)
        else:
            return np.random.uniform(minv, maxv)  # floats
    else:  # discrete values
        return np.random.choice(vals)


# ----------------------------------------------------------------------------------------
def get_xshift_bounds(args, xscale, dbgstrs):  # see algebra here https://photos.app.goo.gl/i8jM5Aa8QXvbDD267
    assert args.birth_response == "sigmoid"
    ysc_lo, ysc_hi = args.yscale_range
    br_lo, br_hi = args.initial_birth_rate_range
    lo = (
        math.log(ysc_lo / br_hi - 1.0) / xscale if ysc_lo / br_hi > 1 else -float("inf")
    )
    hi = (
        math.log(ysc_hi / br_lo - 1.0) / xscale if ysc_hi / br_lo > 1 else +float("inf")
    )
    dbgstrs.append("additional xshift bounds from sigmoid/xscale: %.2f  %.2f" % (lo, hi))
    return (lo, hi)


# ----------------------------------------------------------------------------------------
def get_yscale_bounds(args, xscale, xshift, dbgstrs):  # similar to previous fcn
    assert args.birth_response == "sigmoid"
    br_lo, br_hi = args.initial_birth_rate_range
    lo = br_lo * (1 + math.exp(xscale * xshift))
    hi = br_hi * (1 + math.exp(xscale * xshift))
    dbgstrs.append("additional yscale bounds from sigmoid/xscale/xshift: %.2f  %.2f" % (lo, hi))
    return (lo, hi)


# ----------------------------------------------------------------------------------------
def add_pval(pcounts, pname, pval):
    if pname not in pcounts:
        pcounts[pname] = []
    pcounts[pname].append(pval)


# ----------------------------------------------------------------------------------------
def choose_params(args, pcounts, itrial):
    params = {}
    plist = ["xscale", "xshift", "yscale", "time_to_sampling", "carry_cap", "n_seqs"]  # NOTE order of first three has to stay like this (well you'd have to redo the algebra to change the order)
    for pname in plist:  # NOTE compare to loop at end of run_sub_procs()
        extra_bounds, dbgstrs = None, []
        if args.use_generated_parameter_bounds:
            if pname == "xshift":
                extra_bounds = get_xshift_bounds(args, params["xscale"], dbgstrs)
            if pname == "yscale":
                extra_bounds = get_yscale_bounds(args, params["xscale"], params["xshift"], dbgstrs)
        if args.dl_prediction_dir is not None and pname in utils.sigmoid_params:
            if pname == plist[0]:
                print('        choosing dl prediction at index %d (of %d) for: %s' % (itrial % len(args.dl_pvals), len(args.dl_pvals), ' '.join(utils.sigmoid_params)))
            params[pname] = args.dl_pvals[itrial % len(args.dl_pvals)][pname]
        else:
            params[pname] = choose_val(args, pname, extra_bounds=extra_bounds, dbgstrs=dbgstrs)
        if pname in ["time_to_sampling", "carry_cap", "n_seqs"]:
            params[pname] = int(params[pname])
        add_pval(pcounts, pname, params[pname])
    if args.min_survivors is None:
        tfrac = 0.2
        args.min_survivors = tfrac * params['n_seqs']
        print('    setting --min-survivors to %.2f * N seqs = %d' % (tfrac, args.min_survivors))
    if any(params["carry_cap"] < p for p in [args.min_survivors, params["n_seqs"]]):
        print('  %s chose carry cap (%d) smaller than either min survivors %d or N seqs %d, so you\'ll probably either get a lot of failed tree runs, or fail sampling seqs' % (utils.color('yellow', 'warning'), params["carry_cap"], args.min_survivors, params["n_seqs"]))
    pvstrs = ["%s %s" % (p, ("%d" if p in ["time_to_sampling", "carry_cap", "n_seqs"] else "%.2f") % v) for p, v in sorted(params.items())]
    print("    chose new parameter values%s: %s" % ('' if args.simu_bundle_size == 1 else ' (for next bundle of size %d)' % args.simu_bundle_size, "  ".join(pvstrs)))
    return params


# ----------------------------------------------------------------------------------------
def get_responses(args, xscale, xshift, yscale, pcounts):
    # ----------------------------------------------------------------------------------------
    def get_birth():
        if args.birth_response == "constant":
            bresp = ConstantProcess(yscale)
        elif args.birth_response in ["soft-relu", "sigmoid"]:
            if args.birth_response == "sigmoid":
                assert xscale > 0 and yscale > 0, (
                    "xscale and yscale must both be greater than zero for sigmoid response function, but got xscale %.2f, yscale %.2f"
                    % (xscale, yscale)
                )
            kwargs = {
                "xscale": xscale,
                "xshift": xshift,
                "yscale": yscale,
                "yshift": args.yshift,
            }
            rfcns = {
                "soft-relu": poisson.SoftReluProcess,
                "sigmoid": poisson.SigmoidProcess,
            }
            bresp = rfcns[args.birth_response](**kwargs)
        else:
            assert False
        return bresp

    # ----------------------------------------------------------------------------------------
    dresp = ConstantProcess(value=args.death_value)
    bresp = get_birth()
    naive_brate = bresp.λ_homogeneous(NodeState(x=0))
    print("      initial birth rate %.2f (range %s)" % (naive_brate, args.initial_birth_rate_range))
    if naive_brate < args.initial_birth_rate_range[0] - 1e-8 or naive_brate > args.initial_birth_rate_range[1] + 1e-8:
        wstr = 'initial birth response outside specified range: %.3f not in [%.3f, %.3f]' % (naive_brate, args.initial_birth_rate_range[0], args.initial_birth_rate_range[1])
        if args.use_generated_parameter_bounds:
            raise Exception(wstr)
        else:
            print('  %s %s' % (utils.color('yellow', 'warning'), wstr))
    print_resp(bresp, dresp)
    add_pval(pcounts, "initial_birth_rate", naive_brate)

    # if args.debug:
    #     scan_response(bresp, dresp)
    return bresp, dresp


# ----------------------------------------------------------------------------------------
def write_final_outputs(args, all_seqs, all_trees, param_list, inferred=False, dont_encode=False):
    subd = args.tree_inference_method if inferred else ''
    print("  writing %sfinal outputs%s to %s" % ('inferred ' if inferred else '', ' (including internal nodes)' if args.sample_internal_nodes else '', os.path.dirname(outfn(args, "seqs", subd=subd))))

    utils.write_fasta(outfn(args, "seqs", subd=subd), all_seqs)

    with open(outfn(args, "trees", subd=subd), "w") as tfile:
        for pfo in all_trees:
            tfile.write("%s\n" % pfo["tree"].write(format=1))

    lmetafos = []
    for itr, pfo in enumerate(all_trees):
        for node in [pfo["tree"]] + list(pfo["tree"].iter_descendants()):  # both internal and leaf nodes always get written to this file
            lmetafos.append(
                {
                    "tree-index": itr + args.itrial_start,
                    "name": node.name,
                    "affinity": node.state.x,
                    "n_muts": node.total_mutations,
                    "n_muts_aa": node.total_aa_muts,
                    "is_leaf" : node.is_leaf(),
                }
            )
    encode.write_leaf_meta(outfn(args, "meta", subd=subd), lmetafos)

    # encode trees
    if dont_encode:  # UGH
        scale_vals = encode.scale_trees([pfo["tree"] for pfo in all_trees])
        encoded_trees, encoded_fitnesses = None, None
    else:
        scale_vals, encoded_trees = encode.encode_trees([pfo["tree"] for pfo in all_trees])
        _, encoded_fitnesses = encode.encode_trees([pfo["tree"] for pfo in all_trees], mtype='fitness', birth_responses=[pfo["birth-response"] for pfo in all_trees])  # this re-scales the tree, which i think is just a waste of time

    # write summary stats
    if len(param_list) != len(all_trees):
        raise Exception('parameter list %d not same length as trees %d' % (len(param_list), len(all_trees)))
    sstats = []
    for itr, (sval, pfo, params) in enumerate(zip(scale_vals, all_trees, param_list)):
        sstats.append(
            {
                "tree": itr + args.itrial_start,
                "mean_branch_length": sval,
                "total_branch_length": sum(
                    n.dist for n in pfo["tree"].iter_descendants()
                ),
                "carry_cap": params["carry_cap"],
                "time_to_sampling": params["time_to_sampling"],
                # NOTE if you add something here, also add it to encode.sstat_fieldnames
            }
        )
    responses = [
        {k: p["%s-response" % k] for k in ["birth", "death"]} for p in all_trees
    ]

    encode.write_training_files(os.path.dirname(outfn(args, "seqs", subd=subd)), encoded_trees, responses, sstats, encoded_fitnesses=encoded_fitnesses)

# ----------------------------------------------------------------------------------------
def add_seqs(args, all_seqs, itrial, tree):
    sample_nodes = [tree] + list(tree.iter_descendants()) if args.sample_internal_nodes else list(tree.iter_leaves())
    lseqs = []
    for node in sample_nodes:
        sfo = {"name": str(node.name), "seq": node.state.sequence}  # when read from pickle, the node names sometimes end up as integers
        all_seqs.append(sfo)
        if node.is_leaf():
            lseqs.append(sfo)
    return lseqs

# ----------------------------------------------------------------------------------------
def add_tree(all_trees, itrial, pfo):
    all_trees.append(pfo)

# ----------------------------------------------------------------------------------------
def get_inferred_tree(args, params, pfo, gp_map, inf_trees, true_leaf_seqs, itrial, outfix='out', debug=False):
    assert args.tree_inference_method in ['iqtree', 'gctree']
    # ----------------------------------------------------------------------------------------
    def run_method(ofn):
        ifn = '%s/input-seqs.fa' % wkdir
        input_seqs = true_leaf_seqs
        if args.tree_inference_method == 'gctree':
            input_seqs.insert(0, {'name' : 'naive', 'seq' : replay.NAIVE_SEQUENCE})
        utils.write_fasta(ifn, input_seqs)
        if args.tree_inference_method == 'iqtree':
            cmd = '%s/iqtree -asr -s %s -pre %s/%s >%s/log' % (os.path.dirname(os.path.realpath(__file__)), ifn, wkdir, outfix, wkdir)
            if os.path.exists(ofn) and args.overwrite:
                cmd += ' -redo'
        elif args.tree_inference_method == 'gctree':
            mfn = '%s/meta.yaml' % wkdir
            with open(mfn, 'w') as mfile:
                mfo = {'h_frame' : 1, 'h_offset' : 0, 'l_frame' : 1, 'l_offset' : replay.CHAIN_2_START_IDX}
                json.dump(mfo, mfile)
            cmd = '%s/bin/gctree-run.py --infname %s --outdir %s --metafname %s' % (args.partis_dir, ifn, wkdir, mfn)
            #  --fix-multifurcations  turn this on if you want to be able to encode them (but then will need to also have it on for data trees)
        else:
            assert False
        print('    %s %s' % (utils.color('red', 'run'), cmd))
        subprocess.check_call(cmd, shell=True)
    # ----------------------------------------------------------------------------------------
    def read_inferred_seqs():
        if args.tree_inference_method == 'iqtree':
            inf_infos = {}
            with open('%s/%s.state'%(wkdir, outfix)) as afile:
                reader = csv.DictReader(filter(lambda row: row[0]!='#', afile), delimiter=str('\t'))
                for line in reader:
                    node = line['Node']
                    if node not in inf_infos:
                        inf_infos[node] = {}
                    inf_infos[node][int(line['Site'])] = line['State'].replace('-', 'N')  # NOTE this has uncertainty info as well, which atm i'm ignoring
            seq_len = len(true_leaf_seqs[0]['seq'])
            inf_seqfos = []
            for node, nfo in inf_infos.items():
                inf_seqfos.append({'name' : node, 'seq' : ''.join(nfo[i] for i in range(1, seq_len+1))})
        elif args.tree_inference_method == 'gctree':
            inf_seqfos = utils.read_fastx(ofn(wkdir, seqs=True))
        else:
            assert False
        if debug:
            print('          read %d %s inferred ancestral seqs' % (len(inf_seqfos), args.tree_inference_method))
        return inf_seqfos
    # ----------------------------------------------------------------------------------------
    def ofn(wkdir, seqs=False):
        if args.tree_inference_method == 'iqtree':
            return '%s/%s.treefile' % (wkdir, outfix)
        elif args.tree_inference_method == 'gctree':
            return '%s/%s' % (wkdir, 'inferred-seqs.fa' if seqs else 'tree.nwk')
        else:
            assert False
    # ----------------------------------------------------------------------------------------
    assert not args.sample_internal_nodes  # would need to think about whether I want to 1) pass inferred seqs to iqtree and/or 2) read iqtree inferred internal nodes
    if debug:
        print('   %s: getting inferred tree' % (utils.color('blue', args.tree_inference_method)))
    wkdir = '%s/%s/itree-%d' % (args.outdir, args.tree_inference_method, itrial)
    if not os.path.exists(wkdir):
        os.makedirs(wkdir)
    if os.path.exists(ofn(wkdir)) and not args.overwrite:
        print('        %s output file exists, not rerunning: %s' % (args.tree_inference_method, ofn(wkdir)))
    else:
        run_method(ofn(wkdir))
    inf_seqfos = read_inferred_seqs()
    tree = utils.get_etree(fname=ofn(wkdir))
    relabel_nodes(args, tree, itrial, only_internal=True, seqfos=inf_seqfos + true_leaf_seqs)
    import python.treeutils as treeutils
    dtree, new_seqfos = treeutils.get_binary_tree(None, inf_seqfos + true_leaf_seqs, etree=tree)
    inf_seqfos += new_seqfos
    tree = utils.get_etree(treestr=dtree.as_string(schema='newick').strip())
    utils.write_fasta('%s/inf-anc-seqs.fa'%wkdir, inf_seqfos)

    if debug > 1:
        print('            %s tree before scaling:' % args.tree_inference_method)
        utils.print_dtree(tree, extra_str='                ')

    # set .x, .t, .total_mutations, and .total_aa_muts
    all_seqs = {s['name'] : s['seq'] for s in true_leaf_seqs + inf_seqfos}
    all_aa_seqs = {n : utils.ltranslate(s) for n, s in all_seqs.items()}
    hdcache, hdc_aa = {}, {}
    for tnode in [tree] + list(tree.iter_descendants(strategy='preorder')):
        nseq, aa_seq = all_seqs[tnode.name], all_aa_seqs[tnode.name]
        tnode.x = gp_map(nseq)
        tnode.t = tnode.dist + (0 if tnode.is_root() else tnode.up.t)
        if nseq not in hdcache:
            hdcache[nseq] = utils.hamming_distance(all_seqs[tree.name], nseq)
        if aa_seq not in hdc_aa:
            hdc_aa[aa_seq] = utils.hamming_distance(all_aa_seqs[tree.name], aa_seq, amino_acid=True)
        tnode.total_mutations = hdcache[nseq]
        tnode.total_aa_muts = hdc_aa[aa_seq]
    encode.scale_tree(tree, new_mean_depth=params['time_to_sampling'])

    naive_hdist = utils.hamming_distance(replay.NAIVE_SEQUENCE, all_seqs[tree.name])
    if naive_hdist != 0:
        print('              %s inferred root not equal to replay naive seq (%d bases differ)' % (utils.color('yellow', 'note'), naive_hdist))

    if debug:
        print('            after scaling:')
        def dstr(d): return utils.color('blue', '0', width=9) if float(d)==0 else '%9.6f'%d
        print('                               dist        t          x')
        for tnode in [tree] + list(tree.iter_descendants(strategy='preorder')):
            print('            %15s  %s  %s   %s' % (tnode.name, dstr(tnode.dist), dstr(tnode.t), dstr(tnode.state.x)))

    inf_pfo = {'%s-response'%r : pfo['%s-response'%r] for r in ['birth', 'death']}
    inf_pfo['tree'] = tree
    inf_trees.append(inf_pfo)

# ----------------------------------------------------------------------------------------
def read_dill_file(fname):
    pfo = None
    try:
        with open(fname, "rb") as pfile:
            pfo = dill.load(pfile)
    except Exception as ex:
        print(
            "    %s reading pickle file %s:\n            %s"
            % (utils.color("red", "error"), fname, ex)
        )
    return pfo


# ----------------------------------------------------------------------------------------
def check_memory(itrial, max_frac=0.03):
    mfrac = utils.memory_usage_fraction(extra_str="trial %3d:  " % itrial, debug=True)
    if mfrac > max_frac:
        raise Exception(
            "too much memory: %.3f%% > %.3f%%" % (100 * mfrac, 100 * max_frac)
        )
        return True


def get_parser():
    helpstr = """
    Simulate B cell trees in germinal centers using the birth-death-mutation model.
    Example usage that samples parameter values from within ranges:
        gcd-simulate --debug 1 --outdir <outdir> --xscale-range 0.5 5 --xshift-range -0.5 3 --yscale-range 1 50 --initial-birth-rate-range 2 10 --carry-cap-values 150 --time-to-sampling-values 10 --n-trials 1
    Example usage with multiple subprocesses:
        gcd-simulate --debug 1 --outdir <outdir> --carry-cap-values 150 --time-to-sampling-values 10 --n-trials 100 --n-sub-procs 10

    """
    class MultiplyInheritedFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(formatter_class=MultiplyInheritedFormatter, description=helpstr)
    parser.add_argument("--n-seqs-values", default=[70], nargs="+", type=int, help="Number of sequences to observe (list of values)")
    parser.add_argument("--n-seqs-range", nargs="+", type=int)
    parser.add_argument("--n-trials", default=51, type=int, help="Number of trials/GCs to simulate")
    parser.add_argument("--n-max-tries", default=100, type=int, help="Number of times to retry simulation if it fails due to reaching either the min or max number of leaves.")
    parser.add_argument("--time-to-sampling-values", default=[20], nargs="+", type=int, help="List of values from which to choose for time to sampling.")
    parser.add_argument("--time-to-sampling-range", nargs="+", type=int, help="Pair of values (min/max) between which to choose at uniform random the time to sampling for each tree. Overrides --time-to-sampling-values.")
    parser.add_argument("--simu-bundle-size", default=1, type=int, help="By default, we choose a new set of parameters for each tree. If this arg is set, once we've chosen a set of parameter values, we instead simulate this many trees with those same values before choosing another set.")
    parser.add_argument("--min-survivors", type=int, help="The simulation is terminated if the number of leaves falls below this. If not set, it\'s set automatically based on carry capacity.")
    parser.add_argument("--carry-cap-values", default=[300], nargs='+', type=int)
    parser.add_argument("--carry-cap-range", nargs='+', type=int)
    parser.add_argument("--capacity-method", default="birth", choices=["birth", "death", "hard", None], help="see bdms.evolve() docs. Note that 'death' often involves a ton of churn, which makes for very slow simulations.")
    parser.add_argument("--init-population", type=int, default=2)
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--outdir", default=os.getcwd())
    parser.add_argument("--birth-response", default="sigmoid", choices=["constant", "soft-relu", "sigmoid"], help="birth rate response function")
    # parser.add_argument('--birth-value', default=0.5, type=float, help='value (parameter) for constant birth response')
    parser.add_argument("--death-value", default=0.1, type=float, help="value (parameter) for constant death response")
    parser.add_argument("--xscale-values", default=[1], nargs="+", type=float, help="list of birth response xscale parameter values from which to choose")
    parser.add_argument("--xshift-values", default=[1], nargs="+", type=float, help="list of birth response xshift parameter values from which to choose")
    parser.add_argument("--yscale-values", default=[15], nargs="+", type=float, help="list of birth response yscale parameter values from which to choose")
    parser.add_argument("--xscale-range", nargs="+", type=float, help="Pair of values (min/max) between which to choose at uniform random the birth response xscale parameter for each tree. Overrides --xscale-values. Suggest 0.5 5")
    parser.add_argument("--xshift-range", nargs="+", type=float, help="Pair of values (min/max) between which to choose at uniform random the birth response xshift parameter for each tree. Overrides --xshift-values. Suggest -0.5 3")
    parser.add_argument("--yscale-range", nargs="+", type=float, help="Pair of values (min/max) between which to choose at uniform random the birth response yscale parameter for each tree. Overrides --yscale-values. Suggest 1 50")
    parser.add_argument("--initial-birth-rate-range", default=[2, 10], nargs="+", type=float, help="Pair of values (min/max) for initial/default/average growth rate (i.e. when affinity/x=0). Used to set --yscale.")
    parser.add_argument("--yshift", default=0, type=float, help="atm this shouldn't (need to, at least) be changed")
    parser.add_argument("--mutability-multiplier", default=0.68, type=float)
    parser.add_argument('--dl-prediction-dir', help='If set, look for deep learning (dl) predictions in this dir, and simulate using the predicted parameter values therein')
    parser.add_argument("--sample-internal-nodes", action='store_true', help="By default, only sequences for leaf nodes are written to output (although affinities are written for both leaf and internal nodes). If this arg is set, we write seqs also for internal nodes.")
    parser.add_argument("--n-sub-procs", type=int, help="If set, the --n-trials are split among this many sub processes (which are recursively run with this script). Note that in terms of random seeds, results will not be identical with/without --n-sub-procs set (since there's no way to synchronize seeds partway through))")
    parser.add_argument("--n-max-procs", type=int, help="If set (and --n-sub-procs is set), only run this many sub procs at a time (e.g. to conserve memory).")
    parser.add_argument("--dry-run", action='store_true', help="If --n-sub-procs is set, don\'t actually run the subprocess, instead just print the commands that would be run")
    parser.add_argument("--itrial-start", type=int, default=0, help="if running sub procs (--n-sub-procs) set this so each sub proc's trial index starts at the proper value")
    parser.add_argument("--debug", type=int, default=0, help="Verbosity level; set to 1 or 2 for more debug output.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dont-run-new-simu", action="store_true", help="by default, if some trees are already there but others are missing, we try to rerun the missing ones; if this is set we instead ignore any missing ones and just merge any that are there")
    parser.add_argument("--test", action="store_true", help="sets some default parameter values that run quickly and successfully, i.e. useful for quick tests")
    parser.add_argument("--make-plots", action="store_true", help="")
    parser.add_argument("--label-leaf-internal-nodes", action="store_true", help="Instead of the default node naming scheme of pure integers, add a prefix to the integer indicating if it\'s a leaf (\'leaf-\') or internal (\'mrca-\') node.")
    parser.add_argument("--n-to-plot", type=int, default=10, help="number of tree slice plots to make")
    parser.add_argument("--tree-inference-method")
    parser.add_argument("--partis-dir", default='%s/work/partis'%os.getenv('HOME'))
    return parser


# ----------------------------------------------------------------------------------------
def set_test_args(args):
    if "--carry-cap-values" not in sys.argv:
        args.carry_cap_values = [100]
    if "--n-trials" not in sys.argv:
        args.n_trials = 1
    if "--time-to-sampling-values" not in sys.argv:
        args.time_to_sampling_values = [10]
    if "--min-survivors" not in sys.argv:
        args.min_survivors = 10
    if "--n-seqs" not in sys.argv:
        args.n_seqs_values = [5]
    if "--n-max-tries" not in sys.argv:
        args.n_max_tries = 5
    print(
        "  --test: --n-trials %d --carry-cap-values %s  --time-to-sampling %s  --min-survivors %d  --n-seqs %d  --n-max-tries %d"
        % (
            args.n_trials,
            args.carry_cap_values,
            args.time_to_sampling_values,
            args.min_survivors,
            args.n_seqs_values,
            args.n_max_tries,
        )
    )


# ----------------------------------------------------------------------------------------
def plot_params_responses(args, pcounts, all_trees=None, all_fns=None):
    if all_fns is None:
        all_fns = [[]]
    if args.birth_response == "sigmoid":  # could plot other ones, but I think I need to modify some things, and I don't need it atm
        if args.make_plots and all_trees is not None:
            utils.plot_phenotype_response(args.outdir + "/plots/responses", all_trees, bundle_size=args.simu_bundle_size, fnames=all_fns)
    # make summary param plots even if --make-plots isn't set (just cause atm i don't want to add another arg to make some-but-not-all plots)
    utils.plot_chosen_params(args.outdir + "/plots/params", pcounts, {p: getattr(args, p.replace("-", "_") + "_range") for p in pcounts}, fnames=all_fns)
    utils.make_html(args.outdir + "/plots", fnames=all_fns)

    print("    sampled parameter values:               min      max")
    for pname, pvals in sorted(pcounts.items()):
        print("                      %27s  %7.2f  %7.2f" % (pname, min(pvals), max(pvals)))


# ----------------------------------------------------------------------------------------
def run_sub_procs(args):
    # ----------------------------------------------------------------------------------------
    def merge_subproc_outputs(subd=None):
        odr = os.path.dirname(outfn(args, "seqs", subd=subd))
        print("    writing merged %sfiles to %s" % ('' if subd is None else args.tree_inference_method+' ', odr))
        print("        N files  N trees  time (s)  memory %   ftype")
        if not os.path.exists(odr):
            os.makedirs(odr)
        missing_trees = None
        for ftype in encode.final_ofn_strs:
            ofn = outfn(args, ftype, subd=subd)
            fnames = [
                outfn(args, ftype, subd="iproc-%d%s"%(i, '' if subd is None else '/'+subd))
                for i in range(args.n_sub_procs)
            ]
            start = time.time()
            n_total_trees = ''
            if ftype in ["seqs", "trees", "meta", "summary-stats"]:
                if ftype in ["meta", "summary-stats"]:
                    cmds = [
                        "head -n1 %s >%s" % (fnames[0], ofn),
                        "tail --quiet -n+2 %s >>%s" % (" ".join(fnames), ofn),
                    ]
                else:
                    cmds = ["cat %s >%s" % (" ".join(fnames), ofn)]
                for cmd in cmds:
                    subprocess.check_call(cmd, shell=True)
                if ftype == 'trees':
                    n_total_trees = int(subprocess.check_output('wc -l %s | cut -d\' \' -f1' % ofn, shell=True))
            elif ftype in ["encoded-trees", "encoded-fitnesses"]:
                elists = [encode.read_trees(fn) for fn in fnames]
                n_total_trees = sum(len(l) for l in elists)
                missing_trees = [n_per_proc - len(l) for l in elists]
                all_etrees = [e for fn in fnames for e in encode.read_trees(fn)]
                encode.write_trees(ofn, all_etrees, 'tree' if 'tree' in ftype else 'fitness')
            elif ftype in ["responses"]:
                all_responses = []
                for fn in fnames:
                    with open(fn, "rb") as rfile:
                        all_responses += pickle.load(rfile)
                n_total_trees = len(all_responses)
                with open(ofn, "wb") as rfile:
                    dill.dump(all_responses, rfile)
            else:
                raise Exception("unexpected file type %s" % ftype)
            print(
                "         %3d %9s    %5.2f   %7.2f      %s"
                % (
                    len(fnames),
                    str(n_total_trees),
                    time.time() - start,
                    100 * utils.memory_usage_fraction(),
                    ftype,
                )
            )
        if any(m > 0 for m in missing_trees):
            print('    %s missing %d total trees over %d procs: %s' % (utils.color('yellow', 'warning'), sum(missing_trees), args.n_sub_procs, ' '.join(utils.color('red' if m>0 else None, str(m)) for m in missing_trees)))
    # ----------------------------------------------------------------------------------------
    procs = []
    if args.n_trials % args.n_sub_procs != 0:
        raise Exception("--n-trials %d has to be divisible by --n-sub-procs %d, but got remainder %d (otherwise it's too easy to run into issues with bundling)" % (args.n_trials, args.n_sub_procs, args.n_trials % args.n_sub_procs))
    n_per_proc = int(args.n_trials / float(args.n_sub_procs))
    print("    starting %d procs with %d events per proc" % (args.n_sub_procs, n_per_proc))
    if args.simu_bundle_size != 1:  # make sure that all chunks of trees with same parameters are of same length, i.e. that last chunk isn't smaller (especially important if this is a subproc whose output will be smashed together with others)
        if n_per_proc % args.simu_bundle_size != 0:
            raise Exception("N trees per proc %d ( = --n-trials / --n-sub-procs = %d / %d) has to be evenly divisible by --simu-bundle-size %d, but got remainder %d" % (n_per_proc, args.n_trials, args.n_sub_procs, args.simu_bundle_size, n_per_proc % args.simu_bundle_size))
        print("      making bundles of %d trees for each set of parameter values (%d bundles per sub proc)" % (args.simu_bundle_size, n_per_proc / args.simu_bundle_size))
    for iproc in range(args.n_sub_procs):
        clist = ["python"] + copy.deepcopy(sys.argv)
        subdir = "%s/iproc-%d" % (args.outdir, iproc)
        istart = iproc * n_per_proc
        if (
            all(
                os.path.exists(outfn(args, ft, subd='iproc-%d'%iproc))
                for ft in encode.final_ofn_strs
            )
            and not args.overwrite
        ):
            print("        proc %d: final outputs exist" % iproc)
            sys.stdout.flush()
            continue
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        utils.replace_in_arglist(clist, "--outdir", subdir)
        utils.replace_in_arglist(clist, "--seed", str(args.seed + istart))
        utils.replace_in_arglist(clist, "--n-trials", str(istart + n_per_proc))
        utils.replace_in_arglist(
            clist,
            "--itrial-start",
            str(istart),
            insert_after="--n-trials",
            has_arg=True,
        )
        utils.remove_from_arglist(clist, "--n-sub-procs", has_arg=True)
        cmd_str = " ".join(clist)
        print("      %s %s" % (utils.color("red", "run"), cmd_str))
        sys.stdout.flush()
        if args.dry_run:
            continue
        logfname = "%s/simu.log" % subdir
        if os.path.exists(logfname):
            subprocess.check_call(
                (
                    "mv %s %s.old.%d" % (logfname, logfname, random.randint(100, 999))
                ).split()
            )  # can't be bothered to iterate properly like in the partis code
        subprocess.check_call("echo %s >%s" % (cmd_str, logfname), shell=True)
        cmd_str = "%s >>%s" % (cmd_str, logfname)
        procs.append(subprocess.Popen(cmd_str, env=os.environ, shell=True))
        if args.n_max_procs is not None:
            utils.limit_procs(procs, args.n_max_procs)
    if args.dry_run:
        print('  --dry-run: exiting')
        return
    while procs.count(None) != len(procs):  # we set each proc to None when it finishes
        for iproc in range(len(procs)):
            if procs[iproc] is None:  # already finished
                continue
            if procs[iproc].poll() is not None:  # it just finished
                procs[iproc].communicate()
                if procs[iproc].returncode == 0:
                    procs[iproc] = None  # job succeeded
                    print("    proc %d finished" % iproc)
                else:
                    raise Exception(
                        "process %d failed with exit code %d"
                        % (iproc, procs[iproc].returncode)
                    )
        sys.stdout.flush()
        time.sleep(0.01 / max(1, len(procs)))
    merge_subproc_outputs()
    if args.tree_inference_method is not None:
        merge_subproc_outputs(subd=args.tree_inference_method)

    if args.make_plots:
        print("  note: can't make per-tree plots in main process when --n-sub-procs is set")

    # read merged files so we can make parameter count plots
    with open(outfn(args, 'responses'), "rb") as rfile:
        pklfos = pickle.load(rfile)
    pcounts = {}
    for pkfo in pklfos:
        for pname, pval in pkfo['birth']._param_dict.items():  # NOTE compare to loop in choose_params()
            if pname == 'yshift':  # not varying this atm (and maybe not ever)
                continue
            add_pval(pcounts, pname, pval)
        add_pval(pcounts, "initial_birth_rate", pkfo['birth'].λ_homogeneous(NodeState(x=0)))
    with open(outfn(args, 'summary-stats')) as cfile:
        reader = csv.DictReader(cfile)
        for line in reader:
            for pname in [p for p in ['carry_cap', 'time_to_sampling', 'n_seqs'] if p in line]:  # old files don't have these
                add_pval(pcounts, pname, float(line[pname]))
    plot_params_responses(args, pcounts)


# ----------------------------------------------------------------------------------------
def main():
    git_dir = os.path.dirname(os.path.realpath(__file__)).replace("/scripts", "/.git")
    print("    gcdyn commit: %s" % subprocess.check_output(["git", "--git-dir", git_dir, "rev-parse", "HEAD"]).strip())
    parser = get_parser()
    args = parser.parse_args()
    if args.init_population < 2 or (args.init_population & (args.init_population-1) != 0):
        raise Exception('--init-population must be a positive power of 2, but got %d' % args.init_population)
    if args.simu_bundle_size != 1 and args.n_trials % args.simu_bundle_size != 0:
        raise Exception("--n-trials %d not evenly divisible by --simu-bundle-size %d" % (args.n_trials, args.simu_bundle_size))
    args.use_generated_parameter_bounds = args.birth_response == "sigmoid" and None not in [args.yscale_range, args.initial_birth_rate_range]  # if either yscale or initial birth rate have no specified range, we can't calculate xshift and yscale ranges (well maybe could do one, but generally if you want ranges, specify them)
    if args.use_generated_parameter_bounds:
        print("    using additional generated parameter bounds")
    else:
        print("  note: not using additional generated parameter bounds since at least one of --yscale-range, --initial-birth-rate-range was unset (this may result in lots of failed simulation runs if the initial birth rate is either too small or too large)")
    # handle args that can have either a list of a few values, or choose from a uniform interval specified with two (min, max) values
    for pname in ["xscale", "xshift", "yscale", "time_to_sampling", "carry_cap", "n_seqs"]:
        rangevals = getattr(args, pname + "_range")
        if rangevals is not None and len(rangevals) != 2:  # range with two values for continuous
            raise Exception("range must consist of two values but got %d" % len(rangevals))
    if args.dl_prediction_dir is not None:
        prfn = '%s/test.csv' % args.dl_prediction_dir
        print('  reading dl prediction from %s' % prfn)
        args.dl_pvals = []
        with open(prfn) as cfile:
            reader = csv.DictReader(cfile)
            for line in reader:
                args.dl_pvals.append({p : float(line['%s-predicted'%p]) for p in utils.sigmoid_params})
    if args.tree_inference_method is not None and args.sample_internal_nodes:
        raise Exception('this isn\'t implemented, and it\'s not really clear that it should be -- for instance, do the internal nodes get passed to tree inference, or do we just sample the inferred internal nodes?')

    random.seed(args.seed)
    np.random.seed(args.seed)
    np.seterr(divide="ignore")

    start = time.time()
    if args.test:
        set_test_args(args)
    if args.n_sub_procs is not None:
        run_sub_procs(args)
        print("    total simulation time: %.1f sec" % (time.time() - start))
        sys.exit(0)

    assert args.death_value >= 0

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if args.make_plots:
        for sfn in glob.glob("%s/plots/tree-slices/*.svg" % args.outdir):
            os.remove(sfn)

    dmsfo = replay.dms(debug=True)
    gp_map = gpmap.AdditiveGPMap(
        dmsfo["affinity"], nonsense_phenotype=dmsfo["affinity"].min().min()
    )
    assert gp_map(replay.NAIVE_SEQUENCE) == 0

    mutator = mutators.SequencePhenotypeMutator(
        mutators.ContextMutator(
            mutability=args.mutability_multiplier * replay.mutability(),
            substitution=replay.substitution(),
        ),
        gp_map,
    )

    mutation_resp = poisson.SequenceContextMutationProcess(
        args.mutability_multiplier * replay.mutability(),
    )

    all_seqs, all_trees, inf_trees = [], [], []
    n_missing = 0
    rng = np.random.default_rng(seed=args.seed)
    params, n_times_used, pcounts, plist = None, 0, {}, []  # parameter values, and number of trees that we've simulated with these parameter values (pcounts keeps track of counts of each parameter (for summary printing/plots) whereas plist keeps all parameter values so we can write summary stats at the end)
    all_fns = [[]]  # just for plotting
    for itrial in range(args.itrial_start, args.n_trials):
        print(utils.color("blue", "trial %d:" % itrial), end=" ")
        check_memory(itrial)
        ofn = outfn(args, None, itrial=itrial)
        if params is None or n_times_used == args.simu_bundle_size:  # first time through loop or start of a new bundle
            params = choose_params(args, pcounts, itrial)  # NOTE make *sure* you always get to this point in the loop (i.e. don't put any continue statements above it)
            n_times_used = 0
        n_times_used += 1
        if os.path.exists(ofn) and not args.overwrite:
            print("    output %s already exists, skipping" % ofn)
            pfo = read_dill_file(ofn)
            if args.make_plots:
                print("    note: can't make N vs time plots when reading pickle files since we write pruned trees (i.e. you need to rm/overwrite to actually rerun the simulation)")
            if pfo is None:  # file is screwed up and we want to rerun
                print("    rerunning since pickle file read failed")
            else:
                tree_leaf_seqs = add_seqs(args, all_seqs, itrial, pfo["tree"])
                add_tree(all_trees, itrial, pfo)
                plist.append(params)
                if args.tree_inference_method is not None:
                    get_inferred_tree(args, params, pfo, gp_map, inf_trees, tree_leaf_seqs, itrial, debug=args.debug)
                continue
        if args.dont_run_new_simu:
            n_missing += 1
            continue
        sys.stdout.flush()
        birth_resp, death_resp = get_responses(args, params["xscale"], params["xshift"], params["yscale"], pcounts)
        fnlist, tree = generate_sequences_and_tree(
            args,
            params,
            birth_resp,
            death_resp,
            mutation_resp,
            mutator,
            gp_map,
            itrial,
            seed=rng,
        )
        if tree is None:
            n_missing += 1
            continue
        if fnlist is not None:
            for fn in fnlist:
                utils.addfn(all_fns, fn)
        with open(ofn, "wb") as fp:
            dill.dump({"tree": tree, "birth-response": birth_resp, "death-response": death_resp}, fp)
        tree_leaf_seqs = add_seqs(args, all_seqs, itrial, tree)
        pfo = {"tree": tree, "birth-response": birth_resp, "death-response": death_resp}
        add_tree(all_trees, itrial, pfo)
        plist.append(params)
        if args.tree_inference_method is not None:
            get_inferred_tree(args, params, pfo, gp_map, inf_trees, tree_leaf_seqs, itrial, debug=args.debug)

    if args.dont_run_new_simu:
        print("    --dont-run-new-simu: missing %d trees, but ignoring and just merging the ones we have" % n_missing)
    if n_missing > 0:
        print("    %s missing %d / %d trees (it's generally expected that some will fail, so this is probably ok if it's not too many)" % (utils.color("yellow", "warning"), n_missing, args.n_trials - args.itrial_start))
    if len(all_trees) == 0:
        print("  %s no resulting trees, exiting without writing or plotting anything" % utils.color("yellow", "warning"))
        sys.exit(0)

    write_final_outputs(args, all_seqs, all_trees, plist)
    if args.tree_inference_method is not None:
        write_final_outputs(args, all_seqs, inf_trees, plist, inferred=True, dont_encode=args.tree_inference_method=='gctree')  # need to turn on --fix-multifurcations above if i want to encode gctrees (atm just want to make replay comparison plots, which doesn't require encoding)

    plot_params_responses(args, pcounts, all_trees=all_trees, all_fns=all_fns)
    print("    total simulation time: %.1f sec" % (time.time() - start))
# fmt: on
