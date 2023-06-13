import numpy as np
import argparse
import os
import sys

# import colored_traceback.always  # need to add this to installation stuff, i'm not sure how to do it atm
import dill
import time
import copy
import random
import subprocess

from gcdyn import bdms, gpmap, mutators, poisson, utils, encode
from experiments import replay
from colors import color


# ----------------------------------------------------------------------------------------
def outfn(ftype, itrial, odir=None):
    assert ftype in ["fasta", "nwk", "pkl", "npy"]
    if odir is None:
        odir = args.outdir
    if itrial is None:
        ftstrs = {
            "fasta": "seqs",
            "nwk": "trees",
            "npy": "encoded-trees",
            "pkl": "responses",
        }
        tstr = ftstrs.get(ftype, "simu")
    else:
        tstr = "tree_%d" % itrial
    return f"{odir}/{tstr}.{ftype}"


# ----------------------------------------------------------------------------------------
def print_final_response_vals(tree, birth_resp, death_resp):
    print("                           x         birth           death")
    print("      time   N seqs.   min   max    min  max       min     max")
    xvals, bvals, dvals = [], [], []
    for tval in range(
        args.time_to_sampling + 1
    ):  # kind of weird/arbitrary to take integer values
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
    birth_resp,
    death_resp,
    mutation_rate,
    mutator,
    seed=0,
):
    err_strs, success = {}, False
    for iter in range(args.n_max_tries):
        try:
            tree_start = time.time()
            tree = bdms.TreeNode()
            tree.x = gp_map(replay.NAIVE_SEQUENCE)
            tree.sequence = replay.NAIVE_SEQUENCE
            tree.sequence_context = replay.seq_to_contexts(replay.NAIVE_SEQUENCE)
            tree.evolve(
                args.time_to_sampling,
                birth_response=birth_resp,
                death_response=death_resp,
                mutation_response=mutation_rate,
                mutator=mutator,
                min_survivors=args.min_survivors,
                birth_mutations=False,
                capacity=args.carry_cap,
                capacity_method=args.capacity_method,
                seed=seed,
                # verbose=True,
            )
            print(
                "    try %d succeeded, tip count %d  (%.1fs)"
                % (iter + 1, len(tree), time.time() - tree_start)
            )
            if args.debug:
                print_final_response_vals(tree, birth_resp, death_resp)
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
            % (color("yellow", "warning"), err_strs[estr], estr)
        )
    if not success:
        print(
            "  %s exceeded maximum number of tries %d so giving up"
            % (color("yellow", "warning"), args.n_max_tries)
        )
        return None

    n_to_sample = args.n_seqs
    if len(tree) < n_to_sample:
        print(
            "  %s --n-seqs set to %d but tree has only %d tips, so just sampling all of them"
            % (color("yellow", "warning"), n_to_sample, len(tree))
        )
        n_to_sample = len(tree)
    tree.sample_survivors(n=n_to_sample, seed=seed)
    tree.prune()
    tree.remove_mutation_events()

    # check that node sequences and sequence contexts are consistent
    for node in tree.traverse():
        for a, b in zip(replay.seq_to_contexts(node.sequence), node.sequence_context):
            assert a == b
    # check that node times and branch lengths are consistent
    for node in tree.iter_descendants():
        assert np.isclose(node.t - node.up.t, node.dist)

    # delete the sequence contexts since they make the pickle files six times bigger
    for node in tree.traverse(strategy="postorder"):
        delattr(node, "sequence_context")

    return tree


# ----------------------------------------------------------------------------------------
def get_mut_stats(tree):
    tree.total_mutations = 0

    for node in tree.iter_descendants(strategy="preorder"):
        node.total_mutations = node.n_mutations + node.up.total_mutations

    tmut_tots = [leaf.total_mutations for leaf in tree.iter_leaves()]
    return np.mean(tmut_tots), min(tmut_tots), max(tmut_tots)


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
    print("   response    f(x=0)    function")
    for rname, rfcn in zip(["birth", "death"], [bresp, dresp]):
        print("     %s   %7.3f      %s" % (rname, rfcn.λ_phenotype(0), rfcn))


# ----------------------------------------------------------------------------------------
def get_responses(xscale, xshift):
    # ----------------------------------------------------------------------------------------
    def get_birth(yscl):
        if args.birth_response == "constant":
            bresp = poisson.ConstantResponse(yscl)
        elif args.birth_response in ["soft-relu", "sigmoid"]:
            kwargs = {
                "xscale": xscale,
                "xshift": xshift,
                "yscale": yscl,
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

    bresp = get_birth(
        args.yscale
    )  # have to set it once so we can get the value at x=0 (at least i don't know how else to do it)
    yscl = args.yscale * args.initial_birth_rate / bresp.λ_phenotype(0)
    print(
        "    setting yscale so resp(x=0) equals --inital-birth-rate: <yscale> * <--initial-birth-rate>/<birth rate at x=0> = %.2f * %.2f / %.4f = %.2f"
        % (args.yscale, args.initial_birth_rate, bresp.λ_phenotype(0), yscl)
    )
    bresp = get_birth(yscl)
    print_resp(bresp, dresp)

    # if args.debug:
    #     scan_response(bresp, dresp)
    return bresp, dresp


# ----------------------------------------------------------------------------------------
def write_final_outputs(all_seqs, all_trees):
    print("  writing all seqs to %s" % outfn("fasta", None))
    with open(outfn("fasta", None), "w") as asfile:
        for sfo in all_seqs:
            asfile.write(">%s\n%s\n" % (sfo["name"], sfo["seq"]))

    print("  writing all trees to %s" % outfn("nwk", None))
    with open(outfn("nwk", None), "w") as tfile:
        for pfo in all_trees:
            tfile.write("%s\n" % pfo["tree"].write(format=1))

    encoded_trees = []
    for pfo in all_trees:
        encoded_trees.append(encode.encode_tree(pfo["tree"]))
    print("  writing %d encoded trees to %s" % (len(all_trees), outfn("npy", None)))
    encode.write_trees(outfn("npy", None), encoded_trees)

    print(
        "  writing %d trees and birth/death responses to %s"
        % (len(all_trees), outfn("pkl", None))
    )
    with open(outfn("pkl", None), "wb") as pfile:
        dill.dump(
            [{k: p["%s-response" % k] for k in ["birth", "death"]} for p in all_trees],
            pfile,
        )


# ----------------------------------------------------------------------------------------
def add_seqs(all_seqs, itrial, tree):
    def getname(nstr):
        return nstr if itrial is None else "%d-%s" % (itrial, nstr)

    all_seqs.append(
        {
            "name": "naive",
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
        print("    %s reading pickle file %s:\n            %s" % (color("red", "error"), fname, ex))
    return pfo


# ----------------------------------------------------------------------------------------
def check_memory(max_frac=0.05):
    mfrac = utils.memory_usage_fraction(extra_str='trial %3d:  '%itrial, debug=True)
    if mfrac > max_frac:
        raise Exception('too much memory: %.3f%% > %.3f%%' % (100 * mfrac, 100 * max_frac))
        return True

# ----------------------------------------------------------------------------------------
git_dir = os.path.dirname(os.path.realpath(__file__)).replace("/scripts", "/.git")
print(
    "    gcdyn commit: %s"
    % subprocess.check_output(
        ["git", "--git-dir", git_dir, "rev-parse", "HEAD"]
    ).strip()
)

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
parser.add_argument("--time-to-sampling", default=20, type=int)
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
    default=0.025,
    type=float,
    help="value (parameter) for constant death response",
)
parser.add_argument(
    "--xscale-list",
    default=[2],
    type=float,
    nargs="+",
    help="parameters (see also {x,y}{scale,shift}) for birth (constant, sigmoid and soft-relu) response. If more than one is specified, one is chosen at random for each tree.",
)
parser.add_argument("--xshift-list", default=[-2.5], type=float, nargs="+")
parser.add_argument(
    "--yscale",
    default=1,
    type=float,
    help="probably don't want to set this, it's rescaled automatically below",
)
parser.add_argument(
    "--yshift",
    default=0,
    type=float,
    help="atm this shouldn't (need to, at least) be changed",
)
parser.add_argument(
    "--initial-birth-rate",
    default=5.0,
    type=float,
    help="initial/default/average growth rate (i.e. when affinity/x=0). Used to set --yscale.",
)
parser.add_argument("--mutability-multiplier", default=0.68, type=float)
parser.add_argument(
    "--n-sub-procs",
    type=int,
    help="If set, the --n-trials are split among this many sub processes (which are recursively run with this script). Note that in terms of random seeds, results will not be identical with/without --n-sub-procs set (since there's no way to synchronize seeds partway through))",
)
parser.add_argument(
    "--itrial-start",
    type=int,
    default=0,
    help="if running sub procs (--n-sub-procs) set this so each sub proc's trial index starts at the proper value",
)
parser.add_argument("--debug", action="store_true")
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

args = parser.parse_args()

start = time.time()
if args.test:
    if "--carry-cap" not in sys.argv:
        args.carry_cap = 100
    if "--n-trials" not in sys.argv:
        args.n_trials = 1
    if "--time-to-sampling" not in sys.argv:
        args.time_to_sampling = 10
    if "--min-survivors" not in sys.argv:
        args.min_survivors = 10
    if "--n-seqs" not in sys.argv:
        args.n_seqs = 5
    if "--n-max-tries" not in sys.argv:
        args.n_max_tries = 5
    print(
        "  --test: --carry-cap %d --n-trials %d  --time-to-sampling %d  --min-survivors %d  --n-seqs %d  --n-max-tries %d"
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
    n_per_proc = int(args.n_trials / float(args.n_sub_procs))
    for iproc in range(args.n_sub_procs):
        clist = ["python"] + copy.deepcopy(sys.argv)
        subdir = "%s/iproc-%d" % (args.outdir, iproc)
        istart = iproc * n_per_proc
        if all(
            os.path.exists(outfn("pkl", i, odir=subdir))
            for i in range(istart, istart + n_per_proc)
        ):
            print("    proc %d: all outputs exist" % iproc)
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
        print("  %s %s" % (color("red", "run"), cmd_str))
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
    all_seqs, all_trees = [], []
    for iproc in range(args.n_sub_procs):
        subdir = "%s/iproc-%d" % (args.outdir, iproc)
        istart = iproc * n_per_proc
        for itrial in range(istart, istart + n_per_proc):
            pfo = read_dill_file(outfn("pkl", itrial, odir=subdir))
            if pfo is None:
                print(
                    "    %s can't rerun here (delete file by hand)"
                    % color("red", "error")
                )
                continue
            add_seqs(all_seqs, itrial, pfo["tree"])
            add_tree(all_trees, itrial, pfo)
    write_final_outputs(all_seqs, all_trees)
    sys.exit(0)

assert args.death_value >= 0
if args.birth_response == "sigmoid":
    assert (
        all(x > 0 for x in args.xscale_list) and args.yscale > 0
    )  # necessary so that the phenotype increases with phenotype (e.g. affinity)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

gp_map = gpmap.AdditiveGPMap(
    replay.dms()["affinity"], nonsense_phenotype=replay.dms()["affinity"].min().min()
)
assert gp_map(replay.NAIVE_SEQUENCE) == 0

mutator = mutators.SequencePhenotypeMutator(
    mutators.ContextMutator(
        mutability=args.mutability_multiplier * replay.mutability(),
        substitution=replay.substitution(),
    ),
    gp_map,
)

mutation_rate = poisson.SequenceContextMutationResponse(
    args.mutability_multiplier * replay.mutability(),
)

all_seqs, all_trees = [], []
n_missing = 0
rng = np.random.default_rng(seed=args.seed)
for itrial in range(args.itrial_start, args.n_trials):
    check_memory()
    ofn = outfn("pkl", itrial)
    if os.path.exists(ofn) and not args.overwrite:
        print("    output %s already exists, skipping" % ofn)
        pfo = read_dill_file(ofn)
        if pfo is None:  # file is screwed  up and we want to rerun
            print("    rerunning")
        else:
            add_seqs(all_seqs, itrial, pfo["tree"])
            add_tree(all_trees, itrial, pfo)
            continue
    if args.dont_run_new_simu:
        n_missing += 1
        continue
    sys.stdout.flush()
    birth_resp, death_resp = get_responses(
        np.random.choice(args.xscale_list), np.random.choice(args.xshift_list)
    )
    print(color("blue", "trial %d:" % itrial), end=" ")
    tree = generate_sequences_and_tree(
        birth_resp, death_resp, mutation_rate, mutator, seed=rng
    )
    if tree is None:
        n_missing += 1
        continue

    tmean, tmin, tmax = get_mut_stats(tree)
    print(
        "   mutations per sequence:  mean %.1f   min %.1f  max %.1f"
        % (tmean, tmin, tmax)
    )

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
        % (color("yellow", "warning"), n_missing, args.n_trials)
    )

if len(args.xscale_list) > 0:
    xscales = [p["birth-response"].xscale for p in all_trees]
    print(
        "  chosen xscale values: %s"
        % (",  ".join("%.1f: %d" % (x, xscales.count(x)) for x in sorted(set(xscales))))
    )

write_final_outputs(all_seqs, all_trees)

print("    total simulation time: %.1f sec" % (time.time() - start))
