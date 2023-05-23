import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from Bio import SeqIO
# import colored_traceback.always  # need to add this to installation stuff, i'm not sure how to do it atm
import dill
import time
import copy

from gcdyn import bdms, gpmap, mutators, poisson, utils
import subprocess
from experiments import replay
from colors import color

# ----------------------------------------------------------------------------------------
def add_seqs(all_seqs, itrial, seqfos):
    for sfo in seqfos:
        all_seqs.append({'name' : '%d-%s'%(itrial, sfo['name']), 'seq' : sfo['seq']})

# ----------------------------------------------------------------------------------------
def outfn(ftype, itrial):
    if ftype == 'fasta':
        return f"{args.outdir}/seqs_{itrial}.fasta"
    elif ftype == 'nwk':
        return f"{args.outdir}/tree_{itrial}.nwk"
    elif ftype == 'pkl':
        return f"{args.outdir}/tree_{itrial}.pkl"
    else:
        assert False

# ----------------------------------------------------------------------------------------
def print_final_response_vals(tree):
    print('                           x         birth           death')
    print('      time   N seqs.   min   max    min  max       min     max')
    xvals, bvals, dvals = [], [], []
    for tval in range(args.time_to_sampling + 1):  # kind of weird/arbitrary to take integer values
        txv = sorted(tree.slice(tval))
        tbv, tdv = [[r.λ_phenotype(x) for x in txv] for r in [birth_rate, death_rate]]
        xvals += txv
        bvals += tbv
        dvals += tdv
        print('      %3d   %4d     %5.2f %5.2f  %5.2f %5.2f   %6.3f %6.3f' % (tval, len(txv), min(txv), max(txv), min(tbv), max(tbv), min(tdv), max(tdv)))
    print('             mean      min       max')
    print('       x   %6.2f    %6.2f    %6.2f' % (np.mean(xvals), min(xvals), max(xvals)))
    print('     birth %6.2f    %6.2f    %6.2f' % (np.mean(bvals), min(bvals), max(bvals)))
    print('     death %7.3f   %7.3f   %7.3f' % (np.mean(dvals), min(dvals), max(dvals)))

# ----------------------------------------------------------------------------------------
def generate_sequences_and_tree(
    birth_rate, death_rate, mutation_rate, mutator, seed=0,
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
                birth_response=birth_rate,
                death_response=death_rate,
                mutation_response=mutation_rate,
                mutator=mutator,
                min_survivors=args.min_survivors,
                birth_mutations=False,
                capacity=args.carry_cap,
                capacity_method=args.capacity_method,
                seed=seed,
                # verbose=True,
            )
            print("    try %d succeeded, tip count %d  (%.1fs)" % (iter + 1, len(tree), time.time()-tree_start))
            if args.debug:
                print_final_response_vals(tree)
            success = True
            break
        except bdms.TreeError as terr:
            estr = terr.value
            if 'min_survivors' in estr:
                estr = 'min survivors too small (less than %d)' % args.min_survivors
            # elif terr.value.find('maximum number of leaves') == 0:
            #     max_fails += 1
            if estr not in err_strs:
                err_strs[estr] = 0
            err_strs[estr] += 1
            print('%s%s' % ('failures: ' if sum(err_strs.values())==1 else '', '.'), end='', flush=True)
            continue
    print()
    for estr in sorted([k for k, v in err_strs.items() if v > 0]):
        print('      %s %d failures with message \'%s\'' % (color('yellow', 'warning'), err_strs[estr], estr))
    if not success:
        print('  %s exceeded maximum number of tries %d so giving up' % (color('yellow', 'warning'), args.n_max_tries))
        return None

    tree.sample_survivors(n=args.n_seqs, seed=seed)
    tree.prune()
    tree.remove_mutation_events()

    # check that node sequences and sequence contexts are consistent
    for node in tree.traverse():
        for a, b in zip(replay.seq_to_contexts(node.sequence), node.sequence_context):
            assert a == b
    # check that node times and branch lengths are consistent
    for node in tree.iter_descendants():
        assert np.isclose(node.t - node.up.t, node.dist)

    return tree

# ----------------------------------------------------------------------------------------
def get_mut_stats(tree):
    tree.total_mutations = 0

    for node in tree.iter_descendants(strategy="preorder"):
        node.total_mutations = node.n_mutations + node.up.total_mutations

    tmut_tots = [leaf.total_mutations for leaf in tree.iter_leaves()]
    return np.mean(tmut_tots), min(tmut_tots), max(tmut_tots)

# ----------------------------------------------------------------------------------------
def scan_response(xmin=-5, xmax=2, nsteps=10):  # print output values of response function
    dx = (xmax-xmin) / nsteps
    xvals = list(np.arange(xmin, 0, dx)) + list(np.arange(0, xmax + dx, dx))
    rvals = [birth_rate.λ_phenotype(x) for x in xvals]
    xstr = '   '.join('%7.2f'%x for x in xvals)
    rstr = '   '.join('%7.2f'%r for r in rvals)
    print('    x:', xstr)
    print('    r:', rstr)

# ----------------------------------------------------------------------------------------
def print_resp():
    print('   response    f(x=0)    function')
    for rname, rfcn in zip(['birth', 'death'], [birth_rate, death_rate]):
        print('     %s   %7.3f      %s' % (rname, rfcn.λ_phenotype(0), rfcn))

# ----------------------------------------------------------------------------------------
def set_responses():
    if args.birth_response == 'constant':
        birth_rate = poisson.ConstantResponse(args.yscale) #args.birth_value)
    elif args.birth_response in ['soft-relu', 'sigmoid']:
        kwargs = {'xscale' : args.xscale, 'xshift' : args.xshift, 'yscale' : args.yscale, 'yshift' : args.yshift}
        rfcns = {'soft-relu' : poisson.SoftReluResponse, 'sigmoid' : poisson.SigmoidResponse}
        birth_rate = rfcns[args.birth_response](**kwargs)
    else:
        assert False
    death_rate = poisson.ConstantResponse(args.death_value)
    return birth_rate, death_rate

# ----------------------------------------------------------------------------------------
git_dir = os.path.dirname(os.path.realpath(__file__)).replace('/scripts', '/.git')
print('    gcdyn commit: %s' % subprocess.check_output(['git', '--git-dir', git_dir, 'rev-parse', 'HEAD']).strip())

parser = argparse.ArgumentParser()
parser.add_argument('--n-seqs', default=70, type=int, help='Number of sequences to observe')
parser.add_argument('--n-trials', default=51, type=int, help='Number of trials/GCs to simulate')
parser.add_argument('--n-max-tries', default=10, type=int, help='Number of times to retry simulation if it fails due to reaching either the min or max number of leaves.')
parser.add_argument('--time-to-sampling', default=20, type=int)
parser.add_argument('--min-survivors', default=100, type=int)
parser.add_argument('--carry-cap', default=300, type=int)
parser.add_argument('--capacity-method', default='death', choices=['birth', 'death', 'hard', None])
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--outdir', default=os.getcwd())
parser.add_argument('--birth-response', default='soft-relu', choices=['constant', 'soft-relu', 'sigmoid'], help='birth rate response function')
# parser.add_argument('--birth-value', default=0.5, type=float, help='value (parameter) for constant birth response')
parser.add_argument('--death-value', default=0.025, type=float, help='value (parameter) for constant death response')
parser.add_argument('--xscale', default=2, type=float, help='parameters (see also {x,y}{scale,shift}) for birth (consant, sigmoid and soft-relu) response')
parser.add_argument('--xshift', default=-2.5, type=float)
parser.add_argument('--yscale', default=1, type=float, help='probably don\'t want to change this')
parser.add_argument('--yshift', default=0, type=float, help='atm this shouldn\'t (need to, at least) be changed')
parser.add_argument('--mutability-multiplier', default=0.68, type=float)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--test', action='store_true', help='sets some default parameter values that run quickly and successfully, i.e. useful for quick tests')

start = time.time()
args = parser.parse_args()

if args.test:
    args.carry_cap = 100
    args.n_trials = 2
    args.time_to_sampling = 10
    args.min_survivors = 10
    args.n_seqs = 5
    args.n_max_tries = 5
    print('  --test: --carry-cap %d --n-trials %d  --time-to-sampling %d  --min-survivors %d  --n-seqs %d  --n-max-tries %d' % (args.carry_cap, args.n_trials, args.time_to_sampling, args.min_survivors, args.n_seqs, args.n_max_tries))

assert args.death_value >= 0
if args.birth_response == 'sigmoid':
    assert args.xscale > 0 and args.yscale > 0  # necessary so that the phenotype increases with phenotype (e.g. affinity)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

gp_map = gpmap.AdditiveGPMap(
    replay.dms()['affinity'], nonsense_phenotype=replay.dms()['affinity'].min().min()
)
assert gp_map(replay.NAIVE_SEQUENCE) == 0

birth_rate, death_rate = set_responses()
print_resp()

if args.debug:
    scan_response()

mutator = mutators.SequencePhenotypeMutator(
    mutators.ContextMutator(
        mutability=args.mutability_multiplier * replay.mutability(),
        substitution=replay.substitution(),
    ),
    gp_map,
)

mutation_rate = poisson.SequenceContextMutationResponse(
    args.mutability_multiplier * replay.mutability(), #replay.seq_to_contexts
)

all_seqs, all_trees = [], []
rng = np.random.default_rng(seed=args.seed)
for itrial in range(1, args.n_trials + 1):
    print(color('blue', "trial %d:"%itrial), end=' ')
    ofn = outfn('fasta', itrial)
    if os.path.exists(ofn) and not args.overwrite:
        print('    output %s already exists, skipping'%ofn)
        records = list(SeqIO.parse(ofn, 'fasta'))
        add_seqs(all_seqs, itrial, [{'name' : rcd.id, 'seq' : rcd.seq} for rcd in records])
        with open(outfn('pkl', itrial), 'rb') as pfile:
            tree = dill.load(pfile)
        all_trees.append(tree)
        continue
    sys.stdout.flush()
    tree = generate_sequences_and_tree(
        birth_rate, death_rate, mutation_rate, mutator, seed=rng
    )
    if tree is None:
        continue

    tmean, tmin, tmax = get_mut_stats(tree)
    print("   mutations per sequence:  mean %.1f   min %.1f  max %.1f" % (tmean, tmin, tmax))

    with open(outfn('nwk', itrial), "w") as fp:
        fp.write(tree.write() + "\n")
    with open(outfn('pkl', itrial), "wb") as fp:
        dill.dump(tree, fp)

    seqdict = utils.write_leaf_sequences_to_fasta(
        tree,
        ofn,
        naive=replay.NAIVE_SEQUENCE,
    )
    add_seqs(all_seqs, itrial, [{'name' : sname, 'seq' : seq} for sname, seq in seqdict.items()])
    renamed_tree = copy.deepcopy(tree)
    for node in renamed_tree.iter_descendants():
        node.name = '%d-%s' % (itrial, node.name)
    all_trees.append(renamed_tree)

asfn = '%s/all-seqs.fasta' % args.outdir
print('  writing all seqs to %s'%asfn)
with open(asfn, 'w') as asfile:
    for sfo in all_seqs:
        asfile.write('>%s\n%s\n' % (sfo['name'], sfo['seq']))

tfn = '%s/all-trees.nwk' % args.outdir
print('  writing all trees to %s'%tfn)
with open(tfn, 'w') as tfile:
    for tree in all_trees:
        tfile.write('%s\n'%tree.write(format=1))

pkfn = '%s/simu.pkl' % args.outdir
print('  writing %d trees and birth/death responses to %s' % (len(all_trees), pkfn))
with open(pkfn, 'wb') as pfile:
    pkfo = {'birth-response' : birth_rate, 'death-response' : death_rate, 'trees' : all_trees}
    dill.dump(pkfo, pfile)

print('    total simulation time: %.1f sec' % (time.time()-start))
