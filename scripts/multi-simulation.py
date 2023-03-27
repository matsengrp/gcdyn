import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from Bio import SeqIO

from gcdyn import bdms, gpmap, mutators, responses, utils
from experiments import replay
from colors import color

# ----------------------------------------------------------------------------------------
def add_seqs(all_seqs, itrial, seqfos):
    for sfo in seqfos:
        all_seqs.append({'name' : '%d-%s'%(itrial, sfo['name']), 'seq' : sfo['seq']})

# ----------------------------------------------------------------------------------------
def n_expected_seqs(rbirth, rdeath, timeval):
    return 2**((rbirth - rdeath) * timeval)

# ----------------------------------------------------------------------------------------
def print_final_response_vals(tree):
    print('                           x         birth           death')
    print('      time   N seqs.   min   max    min  max       min     max')
    xvals, bvals, dvals = [], [], []
    for tval in range(args.time_to_sampling + 1):  # kind of weird/arbitrary to take integer values
        txv = sorted(tree.slice(tval))
        tbv, tdv = [[r.f(x) for x in txv] for r in [birth_rate, death_rate]]
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

    min_fails, max_fails = 0, 0
    for iter in range(args.n_max_tries):
        try:
            tree = bdms.TreeNode()
            tree.sequence = replay.NAIVE_SEQUENCE
            tree.x = gp_map(tree.sequence)
            tree.evolve(
                args.time_to_sampling,
                birth_rate=birth_rate,
                death_rate=death_rate,
                mutation_rate=mutation_rate,
                mutator=mutator,
                min_survivors=args.min_survivors,
                max_leaves=args.max_leaves,
                birth_mutations=False,
                seed=seed,
            )
            print(f"    try {iter + 1} succeeded, tip count: {len(tree)}")
            if args.debug_response_fcn:
                print_final_response_vals(tree)
            break
        except bdms.TreeError as terr:
            if terr.value.find('minimum number of survivors') == 0:
                min_fails += 1
            elif terr.value.find('maximum number of leaves') == 0:
                max_fails += 1
            print('%s%s' % ('failures: ' if min_fails+max_fails==1 else '', '.'), end='')
            continue
    if min_fails > 0:
        print('   %s %d failures dropped below min N survivors %d' % (color('yellow', 'warning'), min_fails, args.min_survivors))
    if max_fails > 0:
        print('   %s %d failures exceeded max N leaves %d' % (color('yellow', 'warning'), max_fails, args.max_leaves))

    tree.sample_survivors(n=args.n_seqs, seed=seed)
    tree.prune()

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
    rvals = [birth_rate.f(x) for x in xvals]
    xstr = '   '.join('%7.2f'%x for x in xvals)
    rstr = '   '.join('%7.2f'%r for r in rvals)
    print('    x:', xstr)
    print('    r:', rstr)

# ----------------------------------------------------------------------------------------
def print_resp():
    print('   response    f(x=0)    function')
    for rname, rfcn in zip(['birth', 'death'], [birth_rate, death_rate]):
        print('     %s   %7.3f      %s' % (rname, rfcn.f(0), rfcn))
    print('   expected population at time %d (x=0): %d' % (args.time_to_sampling, n_expected_seqs(birth_rate.f(0), death_rate.f(0), args.time_to_sampling)))

# ----------------------------------------------------------------------------------------
def set_responses():
    if args.birth_response == 'constant':
        birth_rate = responses.ConstantResponse(args.yscale) #args.birth_value)
    elif args.birth_response in ['soft-relu', 'sigmoid']:
        kwargs = {'xscale' : args.xscale, 'xshift' : args.xshift, 'yscale' : args.yscale, 'yshift' : args.yshift}
        rfcns = {'soft-relu' : responses.SoftReluResponse, 'sigmoid' : responses.SigmoidResponse}
        birth_rate = rfcns[args.birth_response](**kwargs)
    else:
        assert False
    death_rate = responses.ConstantResponse(args.death_value)
    return birth_rate, death_rate

# ----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--n-seqs', default=70, type=int, help='Number of sequences to observe')
parser.add_argument('--n-trials', default=51, type=int, help='Number of trials/GCs to simulate')
parser.add_argument('--n-max-tries', default=1000, type=int, help='Number of times to retry simulation if it fails due to reaching either the min or max number of leaves.')
parser.add_argument('--time-to-sampling', default=20, type=int)
parser.add_argument('--min-survivors', default=100, type=int)
parser.add_argument('--max-leaves', default=3000, type=int)
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--outdir', default=os.getcwd())
parser.add_argument('--birth-response', default='soft-relu', choices=['constant', 'soft-relu', 'sigmoid'], help='birth rate response function')
# parser.add_argument('--birth-value', default=0.5, type=float, help='value (parameter) for constant birth response')
parser.add_argument('--death-value', default=0.025, type=float, help='value (parameter) for constant death response')
parser.add_argument('--xscale', default=2, type=float, help='parameters (see also {x,y}{scale,shift}) for birth (consant, sigmoid and soft-relu) response')
parser.add_argument('--xshift', default=-2.5, type=float)
parser.add_argument('--yscale', default=0.1, type=float, help='mostly don\'t set this by hand -- better to let the auto scaler below change it to get a reasonable growth rate.')
parser.add_argument('--yshift', default=0, type=float, help='atm this shouldn\'t (need to, at least) be changed')
parser.add_argument('--initial-birth-rate', default=0.45, type=float, help='this gives a reasonable growth rate (with death rate 0.025, this gives ~500 seqs at time 20)')
parser.add_argument('--mutability-multiplier', default=0.68, type=float)
parser.add_argument('--debug-response-fcn', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--test', action='store_true', help='sets some default parameter values that run quickly and successfully, i.e. useful for quick tests')

args = parser.parse_args()

if args.test:
    args.n_trials = 2
    args.time_to_sampling = 5
    args.min_survivors = 10
    args.n_seqs = 5

assert args.death_value >= 0
if args.birth_response == 'sigmoid':
    assert args.xscale > 0 and args.yscale > 0  # necessary so that the phenotype increases with phenotype (e.g. affinity)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

gp_map = gpmap.AdditiveGPMap(
    replay.bind_df, nonsense_phenotype=replay.bind_df.min().min()
)
assert gp_map(replay.NAIVE_SEQUENCE) == 0

birth_rate, death_rate = set_responses()
print_resp()

# if necessary, rescale response fcns
n_exp = n_expected_seqs(birth_rate.f(0), death_rate.f(0), args.time_to_sampling)
rescale, fuzz_factor = False, 0.5
if n_exp > fuzz_factor * args.max_leaves:
    print('  expected number of leaves too high (%d > %.2f * %d), so rescaling response function' % (n_exp, fuzz_factor, args.max_leaves))
    rescale = True
if n_exp * fuzz_factor < args.min_survivors:
    print('  expected number of leaves too small (%d * %.2f < %d), so rescaling response function' % (n_exp, fuzz_factor, args.min_survivors))
    rescale = True
if rescale:
    args.yscale *= args.initial_birth_rate / birth_rate.f(0)
    birth_rate, death_rate = set_responses()
    print_resp()

if args.debug_response_fcn:
    scan_response()

mutator = mutators.SequencePhenotypeMutator(
    mutators.ContextMutator(
        mutability=args.mutability_multiplier * replay.mutability,
        substitution=replay.substitution,
        seq_to_contexts=replay.seq_to_contexts,
    ),
    gp_map,
)

mutation_rate = responses.SequenceContextMutationResponse(
    args.mutability_multiplier * replay.mutability, replay.seq_to_contexts
)

all_seqs = []
rng = np.random.default_rng(seed=args.seed)
for itrial in range(1, args.n_trials + 1):
    print(color('blue', "trial %d:"%itrial), end=' ')
    ofn = f"{args.outdir}/seqs_{itrial}.fasta"
    if os.path.exists(ofn) and not args.overwrite:
        print('    output %s already exists, skipping'%ofn)
        records = list(SeqIO.parse(ofn, 'fasta'))
        add_seqs(all_seqs, itrial, [{'name' : rcd.id, 'seq' : rcd.seq} for rcd in records])
        continue
    sys.stdout.flush()
    tree = generate_sequences_and_tree(
        birth_rate, death_rate, mutation_rate, mutator, seed=rng
    )

    tmean, tmin, tmax = get_mut_stats(tree)
    print("   mutations per sequence:  mean %.1f   min %.1f  max %.1f" % (tmean, tmin, tmax))

    with open(f"{args.outdir}/tree_{itrial}.nwk", "w") as fp:
        fp.write(tree.write() + "\n")

    seqdict = utils.write_leaf_sequences_to_fasta(
        tree,
        ofn,
        naive=replay.NAIVE_SEQUENCE,
    )
    add_seqs(all_seqs, itrial, [{'name' : sname, 'seq' : seq} for sname, seq in seqdict.items()])

asfn = '%s/all-seqs.fasta' % args.outdir
print('  writing all seqs to %s'%asfn)
with open(asfn, 'w') as asfile:
    for sfo in all_seqs:
        asfile.write('>%s\n%s\n' % (sfo['name'], sfo['seq']))
