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

from gcdyn import bdms, gpmap, mutators, poisson, utils, encode
from experiments import replay


# ----------------------------------------------------------------------------------------
def outfn(ftype, itrial, odir=None):
    if odir is None:
        odir = args.outdir
    return encode.simfn(odir, ftype, itrial)


# ----------------------------------------------------------------------------------------
def print_final_response_vals(tree, birth_resp, death_resp, final_time):
    print("                           x         birth           death")
    print("      time   N seqs.   min   max    min  max       min     max")
    xvals, bvals, dvals = [], [], []
    for tval in range(final_time + 1):  # kind of weird/arbitrary to take integer values
        txv = sorted(tree.slice(tval))
        tbv, tdv = [[r.λ_phenotype(x) for x in txv] for r in [birth_resp, death_resp]]
        xvals += txv
        bvals += tbv
        dvals += tdv
        print(
            "      %3d   %4d     %5.2f %5.2f  %5.2f %5.2f   %6.3f %6.3f"
            % (
                tval,
                len(txv),
                min(txv),
                max(txv),
                min(tbv),
                max(tbv),
                min(tdv),
                max(tdv),
            )
        )
    print("             mean      min       max")
    print(
        "       x   %6.2f    %6.2f    %6.2f" % (np.mean(xvals), min(xvals), max(xvals))
    )
    print(
        "     birth %6.2f    %6.2f    %6.2f" % (np.mean(bvals), min(bvals), max(bvals))
    )
    print("     death %7.3f   %7.3f   %7.3f" % (np.mean(dvals), min(dvals), max(dvals)))


# ----------------------------------------------------------------------------------------
def generate_sequences_and_tree(
    sample_time,
    birth_resp,
    death_resp,
    mutation_resp,
    mutator,
    itrial,
    seed=0,
):
    err_strs, success = {}, False
    for iter in range(args.n_max_tries):
        try:
            tree_start = time.time()
            tree = bdms.TreeNode()
            tree.x = gp_map(replay.NAIVE_SEQUENCE)
            tree.sequence = replay.NAIVE_SEQUENCE
            tree.chain_2_start_idx = replay.CHAIN_2_START_IDX
            tree.evolve(
                sample_time,
                birth_response=birth_resp,
                death_response=death_resp,
                mutation_response=mutation_resp,
                mutator=mutator,
                min_survivors=args.min_survivors,
                birth_mutations=False,
                capacity=args.carry_cap,
                capacity_method=args.capacity_method,
                seed=seed,
                verbose=args.debug > 1,
            )
            print(
                "    try %d succeeded, tip count %d  (%.1fs)"
                % (iter + 1, len(tree), time.time() - tree_start)
            )
            if args.debug:
                print_final_response_vals(tree, birth_resp, death_resp, sample_time)
            success = True
            break
        except bdms.TreeError as terr:
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
        print(
            "      %s %d failures with message '%s'"
            % (utils.color("yellow", "warning"), err_strs[estr], estr)
        )
    if not success:
        print(
            "  %s exceeded maximum number of tries %d so giving up"
            % (utils.color("yellow", "warning"), args.n_max_tries)
        )
        return None, None

    fn = None
    if args.make_plots:
        fn = utils.plot_tree_slices(
            args.outdir + "/plots/tree-slices", tree, sample_time, itrial
        )

    n_to_sample = args.n_seqs
    if len(tree) < n_to_sample:
        print(
            "  %s --n-seqs set to %d but tree has only %d tips, so just sampling all of them"
            % (utils.color("yellow", "warning"), n_to_sample, len(tree))
        )
        n_to_sample = len(tree)
    tree.sample_survivors(n=n_to_sample, seed=seed)
    tree.prune()
    tree.remove_mutation_events()

    # check that node times and branch lengths are consistent
    for node in tree.iter_descendants():
        assert np.isclose(node.t - node.up.t, node.dist)

    set_mut_stats(tree)

    return fn, tree


# ----------------------------------------------------------------------------------------
def set_mut_stats(tree):
    tree.total_mutations = 0

    for node in tree.iter_descendants(strategy="preorder"):
        node.total_mutations = node.n_mutations + node.up.total_mutations

    tmut_tots = [leaf.total_mutations for leaf in tree.iter_leaves()]

    tmean, tmin, tmax = np.mean(tmut_tots), min(tmut_tots), max(tmut_tots)
    print(
        "   mutations per sequence:  mean %.1f   min %.1f  max %.1f"
        % (tmean, tmin, tmax)
    )


# ----------------------------------------------------------------------------------------
def scan_response(
    birth_resp, death_resp, xmin=-5, xmax=2, nsteps=10
):  # print output values of response function
    dx = (xmax - xmin) / nsteps
    xvals = list(np.arange(xmin, 0, dx)) + list(np.arange(0, xmax + dx, dx))
    rvals = [birth_resp.λ_phenotype(x) for x in xvals]
    xstr = "   ".join("%7.2f" % x for x in xvals)
    rstr = "   ".join("%7.2f" % r for r in rvals)
    print("    x:", xstr)
    print("    r:", rstr)


# ----------------------------------------------------------------------------------------
def print_resp(bresp, dresp):
    print("        response    f(x=0)    function")
    for rname, rfcn in zip(["birth", "death"], [bresp, dresp]):
        print("          %s   %7.3f      %s" % (rname, rfcn.λ_phenotype(0), rfcn))


# ----------------------------------------------------------------------------------------
def choose_val(pname, extra_bounds=None):
    minmax, vals = [getattr(args, pname + "_" + str) for str in ["range", "values"]]
    if minmax is not None:  # range with two values for continuous
        minv, maxv = minmax
        if extra_bounds is not None:
            minv = max(
                minv, extra_bounds[0]
            )  # use the more restrictive (larger lo, smaller hi) values
            maxv = min(maxv, extra_bounds[1])
        print("    choosing %s within [%.2f, %.2f]" % (pname, minv, maxv))
        if pname == "time_to_sampling":
            return np.random.choice(
                range(minv, maxv + 1)
            )  # integers (note that this is inclusive)
        else:
            return np.random.uniform(minv, maxv)  # floats
    else:  # discrete values
        return np.random.choice(vals)


# ----------------------------------------------------------------------------------------
def get_xshift_bounds(
    xscale,
):  # see algebra here https://photos.app.goo.gl/i8jM5Aa8QXvbDD267
    assert args.birth_response == "sigmoid"
    ysc_lo, ysc_hi = args.yscale_range
    br_lo, br_hi = args.initial_birth_rate_range
    lo = (
        math.log(ysc_lo / br_hi - 1.0) / xscale if ysc_lo / br_hi > 1 else -float("inf")
    )
    hi = (
        math.log(ysc_hi / br_lo - 1.0) / xscale if ysc_hi / br_lo > 1 else +float("inf")
    )
    print("        additional xshift bounds from sigmoid/xscale: %.2f  %.2f" % (lo, hi))
    return (lo, hi)


# ----------------------------------------------------------------------------------------
def get_yscale_bounds(xscale, xshift):  # similar to previous fcn
    assert args.birth_response == "sigmoid"
    br_lo, br_hi = args.initial_birth_rate_range
    lo = br_lo * (1 + math.exp(xscale * xshift))
    hi = br_hi * (1 + math.exp(xscale * xshift))
    print(
        "        additional yscale bounds from sigmoid/xscale/xshift: %.2f  %.2f"
        % (lo, hi)
    )
    return (lo, hi)


# ----------------------------------------------------------------------------------------
def add_pval(pname, pval):
    if pname not in param_counters:
        param_counters[pname] = []
    param_counters[pname].append(pval)


# ----------------------------------------------------------------------------------------
def choose_params():
    params = {}
    for pname in [
        "xscale",
        "xshift",
        "yscale",
        "time_to_sampling",
    ]:  # NOTE order of first three has to stay like this (well you'd have to redo the algebra to change the order)
        extra_bounds = None
        if args.birth_response == "sigmoid":
            if pname == "xshift":
                extra_bounds = get_xshift_bounds(params["xscale"])
            if pname == "yscale":
                extra_bounds = get_yscale_bounds(params["xscale"], params["xshift"])
        params[pname] = choose_val(
            pname, extra_bounds=extra_bounds
        )  # get_xshift_bounds(params['xscale']) if pname=='xshift' else None)
        add_pval(pname, params[pname])
    print(
        "    chose new parameter values: %s"
        % "  ".join(
            "%s %s" % (p, ("%d" if p == "time_to_sampling" else "%.2f") % v)
            for p, v in sorted(params.items())
        )
    )
    return params


# ----------------------------------------------------------------------------------------
def get_responses(xscale, xshift, yscale):
    # ----------------------------------------------------------------------------------------
    def get_birth():
        if args.birth_response == "constant":
            bresp = poisson.ConstantResponse(yscale)
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
                "soft-relu": poisson.SoftReluResponse,
                "sigmoid": poisson.SigmoidResponse,
            }
            bresp = rfcns[args.birth_response](**kwargs)
        else:
            assert False
        return bresp

    # ----------------------------------------------------------------------------------------
    dresp = poisson.ConstantResponse(args.death_value)
    bresp = get_birth()
    print(
        "      initial birth rate %.2f (range %s)"
        % (bresp.λ_phenotype(0), args.initial_birth_rate_range)
    )
    if args.birth_response == "sigmoid":
        assert (
            bresp.λ_phenotype(0) > args.initial_birth_rate_range[0] - 1e-8
            and bresp.λ_phenotype(0) < args.initial_birth_rate_range[1] + 1e-8
        )
    print_resp(bresp, dresp)
    add_pval("initial_birth_rate", bresp.λ_phenotype(0))

    # if args.debug:
    #     scan_response(bresp, dresp)
    return bresp, dresp


# ----------------------------------------------------------------------------------------
def write_final_outputs(all_seqs, all_trees):
    print("  writing final outputs to %s" % args.outdir)

    with open(outfn("seqs", None), "w") as asfile:
        for sfo in all_seqs:
            asfile.write(">%s\n%s\n" % (sfo["name"], sfo["seq"]))

    with open(outfn("trees", None), "w") as tfile:
        for pfo in all_trees:
            tfile.write("%s\n" % pfo["tree"].write(format=1))

    with open(outfn("leaf-meta", None), "w") as jfile:
        writer = csv.DictWriter(jfile, ["name", "affinity", "n_muts"])
        writer.writeheader()
        for pfo in all_trees:
            for node in pfo["tree"].iter_descendants():
                writer.writerow(
                    {
                        "name": node.name,
                        "affinity": node.x,
                        "n_muts": node.total_mutations,
                    }
                )

    scale_vals, encoded_trees = encode.encode_trees([pfo["tree"] for pfo in all_trees])
    sstats = []
    for itr, (sval, pfo) in enumerate(zip(scale_vals, all_trees)):
        sstats.append(
            {
                "tree": itr + args.itrial_start,
                "mean_branch_length": sval,
                "total_branch_length": sum(
                    n.dist for n in pfo["tree"].iter_descendants()
                ),
            }
        )
    responses = [
        {k: p["%s-response" % k] for k in ["birth", "death"]} for p in all_trees
    ]
    encode.write_training_files(args.outdir, encoded_trees, responses, sstats)


# ----------------------------------------------------------------------------------------
def add_seqs(all_seqs, itrial, tree):
    def getname(nstr):
        return nstr if itrial is None else "%d-%s" % (itrial, nstr)

    all_seqs.append(
        {
            "name": getname("naive"),
            "seq": replay.NAIVE_SEQUENCE,
        }
    )
    for leaf in tree.iter_leaves():
        all_seqs.append(
            {
                "name": getname(leaf.name),
                "seq": leaf.sequence,
            }
        )


# ----------------------------------------------------------------------------------------
def add_tree(all_trees, itrial, pfo):
    for node in pfo["tree"].iter_descendants():
        node.name = "%d-%s" % (itrial, node.name)
    all_trees.append(pfo)


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
def check_memory(max_frac=0.03):
    mfrac = utils.memory_usage_fraction(extra_str="trial %3d:  " % itrial, debug=True)
    if mfrac > max_frac:
        raise Exception(
            "too much memory: %.3f%% > %.3f%%" % (100 * mfrac, 100 * max_frac)
        )
        return True


def get_parser():  # needed for sphinx docs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-seqs", default=70, type=int, help="Number of sequences to observe"
    )
    parser.add_argument(
        "--n-trials", default=51, type=int, help="Number of trials/GCs to simulate"
    )
    parser.add_argument(
        "--n-max-tries",
        default=10,
        type=int,
        help="Number of times to retry simulation if it fails due to reaching either the min or max number of leaves.",
    )
    parser.add_argument(
        "--time-to-sampling-values",
        default=[20],
        nargs="+",
        type=int,
        help="List of values from which to choose for time to sampling.",
    )
    parser.add_argument(
        "--time-to-sampling-range",
        nargs="+",
        type=int,
        help="Pair of values (min/max) between which to choose at uniform random the time to sampling for each tree. Overrides --time-to-sampling-values.",
    )
    parser.add_argument(
        "--simu-bundle-size",
        default=1,
        type=int,
        help="By default, we choose a new set of parameters for each tree. If this arg is set, once we've chosen a set of parameter values, we simulate this many trees with those values.",
    )
    parser.add_argument("--min-survivors", default=100, type=int)
    parser.add_argument("--carry-cap", default=300, type=int)
    parser.add_argument(
        "--capacity-method",
        default="birth",
        choices=["birth", "death", "hard", None],
        help="see bdms.evolve() docs. Note that 'death' often involves a ton of churn, which makes for very slow simulations.",
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--outdir", default=os.getcwd())
    parser.add_argument(
        "--birth-response",
        default="sigmoid",
        choices=["constant", "soft-relu", "sigmoid"],
        help="birth rate response function",
    )
    # parser.add_argument('--birth-value', default=0.5, type=float, help='value (parameter) for constant birth response')
    parser.add_argument(
        "--death-value",
        default=0.1,
        type=float,
        help="value (parameter) for constant death response",
    )
    parser.add_argument(
        "--xscale-values",
        default=[2],
        nargs="+",
        type=float,
        help="list of birth response xscale parameter values from which to choose",
    )
    parser.add_argument(
        "--xshift-values",
        default=[-2.5],
        nargs="+",
        type=float,
        help="list of birth response xshift parameter values from which to choose",
    )
    parser.add_argument(
        "--yscale-values",
        default=[1],
        nargs="+",
        type=float,
        help="list of birth response yscale parameter values from which to choose",
    )
    parser.add_argument(
        "--xscale-range",
        nargs="+",
        type=float,
        help="Pair of values (min/max) between which to choose at uniform random the birth response xscale parameter for each tree. Overrides --xscale-values.",
    )
    parser.add_argument(
        "--xshift-range",
        nargs="+",
        type=float,
        help="Pair of values (min/max) between which to choose at uniform random the birth response xshift parameter for each tree. Overrides --xshift-values.",
    )
    parser.add_argument(
        "--yscale-range",
        nargs="+",
        type=float,
        help="Pair of values (min/max) between which to choose at uniform random the birth response yscale parameter for each tree. Overrides --yscale-values.",
    )
    parser.add_argument(
        "--initial-birth-rate-range",
        default=[2, 10],
        nargs="+",
        type=float,
        help="Pair of values (min/max) for initial/default/average growth rate (i.e. when affinity/x=0). Used to set --yscale.",
    )
    parser.add_argument(
        "--yshift",
        default=0,
        type=float,
        help="atm this shouldn't (need to, at least) be changed",
    )
    parser.add_argument("--mutability-multiplier", default=0.68, type=float)
    parser.add_argument(
        "--n-sub-procs",
        type=int,
        help="If set, the --n-trials are split among this many sub processes (which are recursively run with this script). Note that in terms of random seeds, results will not be identical with/without --n-sub-procs set (since there's no way to synchronize seeds partway through))",
    )
    parser.add_argument(
        "--n-max-procs",
        type=int,
        help="If set (and --n-sub-procs is set), only run this many sub procs at a time.",
    )
    parser.add_argument(
        "--itrial-start",
        type=int,
        default=0,
        help="if running sub procs (--n-sub-procs) set this so each sub proc's trial index starts at the proper value",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="Verbosity level; set to 1 or 2 for more debug output.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dont-run-new-simu",
        action="store_true",
        help="by default, if some trees are already there but others are missing, we try to rerun the missing ones; if this is set we instead ignore any missing ones and just merge any that are there",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="sets some default parameter values that run quickly and successfully, i.e. useful for quick tests",
    )
    parser.add_argument("--make-plots", action="store_true", help="")
    parser.add_argument(
        "--n-to-plot", type=int, default=10, help="number of tree slice plots to make"
    )
    return parser

# ----------------------------------------------------------------------------------------
if __name__ == 'main':
    git_dir = os.path.dirname(os.path.realpath(__file__)).replace("/scripts", "/.git")
    print(
        "    gcdyn commit: %s"
        % subprocess.check_output(
            ["git", "--git-dir", git_dir, "rev-parse", "HEAD"]
        ).strip()
    )
    parser = get_parser()
    args = parser.parse_args()
    # handle args that can have either a list of a few values, or choose from a uniform interval specified with two (min, max) values
    for pname in ["xscale", "xshift", "yscale", "time_to_sampling"]:
        rangevals = getattr(args, pname + "_range")
        if (
            rangevals is not None and len(rangevals) != 2
        ):  # range with two values for continuous
            raise Exception("range must consist of two values but got %d" % len(rangevals))
    random.seed(args.seed)
    np.random.seed(args.seed)
    np.seterr(divide='ignore')

    start = time.time()
    if args.test:
        if "--carry-cap" not in sys.argv:
            args.carry_cap = 100
        if "--n-trials" not in sys.argv:
            args.n_trials = 1
        if "--time-to-sampling" not in sys.argv:
            args.time_to_sampling = {"vals": 10}
        if "--min-survivors" not in sys.argv:
            args.min_survivors = 10
        if "--n-seqs" not in sys.argv:
            args.n_seqs = 5
        if "--n-max-tries" not in sys.argv:
            args.n_max_tries = 5
        print(
            "  --test: --carry-cap %d --n-trials %d  --time-to-sampling %s  --min-survivors %d  --n-seqs %d  --n-max-tries %d"
            % (
                args.carry_cap,
                args.n_trials,
                args.time_to_sampling,
                args.min_survivors,
                args.n_seqs,
                args.n_max_tries,
            )
        )

    if (
        args.n_sub_procs is not None
    ):  # this stuff is all copied from partis utils.py, gd it this would be like three lines if i could import that
        procs = []
        if args.n_trials % args.n_sub_procs != 0:
            raise Exception(
                "--n-trials %d has to be divisible by --n-sub-procs %d, but got remainder %d (otherwise it's too easy to run into issues with bundling)"
                % (args.n_trials, args.n_sub_procs, args.n_trials % args.n_sub_procs)
            )
        n_per_proc = int(args.n_trials / float(args.n_sub_procs))
        print(
            "    starting %d procs with %d events per proc" % (args.n_sub_procs, n_per_proc)
        )
        if (
            args.simu_bundle_size != 1
        ):  # make sure that all chunks of trees with same parameters are of same length, i.e. that last chunk isn't smaller (especially important if this is a subproc whose output will be smashed together with others)
            if n_per_proc % args.simu_bundle_size != 0:
                raise Exception(
                    "--n-trees-per-param-set %d has to evenly divide N trees per proc %d ( = --n-trials / --n-sub-procs = %d / %d), but got remainder %d"
                    % (
                        args.simu_bundle_size,
                        n_per_proc,
                        args.n_trials,
                        args.n_sub_procs,
                        n_per_proc % args.simu_bundle_size,
                    )
                )
            print(
                "      bundling %d trees per set of parameter values (%d bundles per sub proc)"
                % (args.simu_bundle_size, n_per_proc / args.simu_bundle_size)
            )
        for iproc in range(args.n_sub_procs):
            clist = ["python"] + copy.deepcopy(sys.argv)
            subdir = "%s/iproc-%d" % (args.outdir, iproc)
            istart = iproc * n_per_proc
            if (
                all(
                    os.path.exists(outfn(ft, None, odir=subdir))
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
        print("    writing merged files to %s" % args.outdir)
        print("        N files   time (s)  memory %   ftype")
        for ftype in encode.final_ofn_strs:
            ofn = outfn(ftype, None)
            fnames = [
                outfn(ftype, None, odir="%s/iproc-%d" % (args.outdir, i))
                for i in range(args.n_sub_procs)
            ]
            start = time.time()
            if ftype in ["seqs", "trees", "leaf-meta", "summary-stats"]:
                if ftype in ["leaf-meta", "summary-stats"]:
                    cmds = [
                        "head -n1 %s >%s" % (fnames[0], ofn),
                        "tail --quiet -n+2 %s >>%s" % (" ".join(fnames), ofn),
                    ]
                else:
                    cmds = ["cat %s >%s" % (" ".join(fnames), ofn)]
                for cmd in cmds:
                    subprocess.check_call(cmd, shell=True)
            elif ftype in ["encoded-trees"]:
                all_etrees = [e for fn in fnames for e in encode.read_trees(fn)]
                encode.write_trees(ofn, all_etrees)
            elif ftype in ["responses"]:
                all_responses = []
                for fn in fnames:
                    with open(fn, "rb") as rfile:
                        all_responses += pickle.load(rfile)
                with open(ofn, "wb") as rfile:
                    dill.dump(all_responses, rfile)
            else:
                raise Exception("unexpected file type %s" % ftype)
            print(
                "         %3d      %5.2f   %7.2f      %s"
                % (
                    len(fnames),
                    time.time() - start,
                    100 * utils.memory_usage_fraction(),
                    ftype,
                )
            )
        if args.make_plots:
            print("  note: can't make plots in main process when --n-sub-procs is set")
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

    mutation_resp = poisson.SequenceContextMutationResponse(
        args.mutability_multiplier * replay.mutability(),
    )

    all_seqs, all_trees = [], []
    n_missing = 0
    rng = np.random.default_rng(seed=args.seed)
    params, n_times_used, param_counters = (
        None,
        0,
        {},
    )  # parameter values, and number of trees that we've simulated with these parameter values
    all_fns = [[]]  # just for plotting
    for itrial in range(args.itrial_start, args.n_trials):
        check_memory()
        ofn = outfn(None, itrial)
        if os.path.exists(ofn) and not args.overwrite:
            print("    output %s already exists, skipping" % ofn)
            pfo = read_dill_file(ofn)
            if args.make_plots:
                print(
                    "    note: can't make tree slice plots when reading pickle files (i.e. you need to rm/overwrite to actually rerun the simulation), since we write pruned trees"
                )
            if pfo is None:  # file is screwed up and we want to rerun
                print("    rerunning")
            else:
                add_seqs(all_seqs, itrial, pfo["tree"])
                add_tree(all_trees, itrial, pfo)
                continue
        if args.dont_run_new_simu:
            n_missing += 1
            continue
        sys.stdout.flush()
        if (
            params is None or n_times_used == args.simu_bundle_size
        ):  # first time through loop or start of a new bundle
            params = choose_params()
            n_times_used = 0
        n_times_used += 1
        birth_resp, death_resp = get_responses(
            params["xscale"], params["xshift"], params["yscale"]
        )
        print(utils.color("blue", "trial %d:" % itrial), end=" ")
        fn, tree = generate_sequences_and_tree(
            params["time_to_sampling"],
            birth_resp,
            death_resp,
            mutation_resp,
            mutator,
            itrial,
            seed=rng,
        )
        if tree is None:
            n_missing += 1
            continue
        utils.addfn(all_fns, fn)

        with open(ofn, "wb") as fp:
            dill.dump(
                {"tree": tree, "birth-response": birth_resp, "death-response": death_resp},
                fp,
            )

        add_seqs(
            all_seqs,
            itrial,
            tree,
        )
        add_tree(
            all_trees,
            itrial,
            {
                "tree": tree,
                "birth-response": birth_resp,
                "death-response": death_resp,
            },
        )

    if args.dont_run_new_simu:
        print(
            "    --dont-run-new-simu: missing %d trees, but ignoring and just merging the ones we have"
            % n_missing
        )
    if n_missing > 0:
        print(
            "    %s missing %d / %d trees (it's generally expected that some will fail, so this is probably ok if it's not too many)"
            % (utils.color("yellow", "warning"), n_missing, args.n_trials)
        )

    write_final_outputs(all_seqs, all_trees)

    if (
        args.make_plots and args.birth_response == "sigmoid"
    ):  # could plot other ones, but I think I need to modify some things, and I don't need it atm
        utils.plot_phenotype_response(
            args.outdir + "/plots/responses",
            all_trees,
            bundle_size=args.simu_bundle_size,
            fnames=all_fns,
        )
        utils.plot_chosen_params(
            args.outdir + "/plots/params",
            param_counters,
            {p: getattr(args, p.replace("-", "_") + "_range") for p in param_counters},
            fnames=all_fns,
        )
        utils.make_html(args.outdir + "/plots", fnames=all_fns)

    print("    sampled parameter values:               min      max")
    for pname, pvals in sorted(param_counters.items()):
        print("                      %17s  %7.2f  %7.2f" % (pname, min(pvals), max(pvals)))

    print("    total simulation time: %.1f sec" % (time.time() - start))
