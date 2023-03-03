import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from gcdyn import bdms, gpmap, mutators, responses, utils
from experiments import replay

# ----------------------------------------------------------------------------------------
def generate_sequences_and_tree(
    birth_rate, death_rate, mutation_rate, mutator, seed=0, time_to_sampling=20
):

    for iter in range(1000):
        try:
            tree = bdms.TreeNode()
            tree.sequence = replay.NAIVE_SEQUENCE
            tree.x = gp_map(tree.sequence)
            tree.evolve(
                time_to_sampling,
                birth_rate=birth_rate,
                death_rate=death_rate,
                mutation_rate=mutation_rate,
                mutator=mutator,
                min_survivors=100,
                max_leaves=1000,
                birth_mutations=False,
                seed=seed,
            )
            print(f"try {iter + 1} succeeded, tip count: {len(tree)}")
            break
        except bdms.TreeError as e:
            print(f"try {iter + 1} failed, {e}", flush=True)
            continue

    tree.sample_survivors(n=args.n_seqs, seed=0)
    tree.prune()

    return tree

# ----------------------------------------------------------------------------------------
def average_mutations(tree):
    tree.total_mutations = 0

    for node in tree.iter_descendants(strategy="preorder"):
        node.total_mutations = node.n_mutations + node.up.total_mutations

    return np.mean([leaf.total_mutations for leaf in tree.iter_leaves()])

# ----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--n-seqs', default=60, type=int, help='Number of sequences to observe')
parser.add_argument('--n-trials', default=10, type=int, help='Number of trials/GCs to simulate')
parser.add_argument('--outdir', default=os.getcwd())
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

gp_map = gpmap.AdditiveGPMap(
    replay.bind_df, nonsense_phenotype=replay.bind_df.min().min()
)
assert gp_map(replay.NAIVE_SEQUENCE) == 0

birth_rate = responses.SoftReluResponse(xscale=2, xshift=-2.5, yscale=0.05, yshift=0)
death_rate = responses.ConstantResponse(0.025)

mutability_multiplier = 1
mutator = mutators.SequencePhenotypeMutator(
    mutators.ContextMutator(
        mutability=mutability_multiplier * replay.mutability,
        substitution=replay.substitution,
        seq_to_contexts=replay.seq_to_contexts,
    ),
    gp_map,
)

mutation_rate = responses.SequenceContextMutationResponse(
    mutability_multiplier * replay.mutability, replay.seq_to_contexts
)

for i in range(1, args.n_trials + 1):
    print("trial #", i)
    rng = np.random.default_rng(seed=i * i)
    tree = generate_sequences_and_tree(
        birth_rate, death_rate, mutation_rate, mutator, seed=rng
    )

    print("average # of mutations per sequence", average_mutations(tree))

    with open(f"{args.outdir}/tree_{i}.nwk", "w") as fp:
        fp.write(tree.write() + "\n")

    utils.write_leaf_sequences_to_fasta(
        tree,
        f"{args.outdir}/seqs_{i}.fasta",
        naive=replay.NAIVE_SEQUENCE,
    )
