import pickle
from collections import defaultdict
from functools import partial

import jax.numpy as np
import pandas as pd
from jax import jit
from jax.config import config
from scipy.stats import gamma, lognorm, norm

from gcdyn import models, mutators, poisson
from gcdyn.mcmc import metropolis_hastings
from gcdyn.utils import sample_trees

config.update("jax_enable_x64", True)

# Tree config

STATE_SPACE = (2, 4, 6, 8)
INITIAL_STATE = 2
PRESENT_TIME = 1.5
NUM_TREES = 15

TRUE_PARAMETERS = {
    "birth_response": poisson.SigmoidResponse(1.0, 5.0, 1.5, 1.0),
    "death_response": poisson.ConstantResponse(1.3),
    "mutation_response": poisson.ConstantResponse(1.3),
    "mutator": mutators.DiscreteMutator(
        state_space=STATE_SPACE,
        transition_matrix=np.array(
            [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
        )
        / 3,
    ),
    "extant_sampling_probability": 1,
    "extinct_sampling_probability": 0,
}

# Sample trees

tree_collections = []

for tree_seed in range(100):
    trees = sample_trees(
        n=NUM_TREES,
        t=PRESENT_TIME,
        init_x=INITIAL_STATE,
        **TRUE_PARAMETERS,
        seed=tree_seed,
    )

    tree_collections.append(trees)

with open("trees.pkl", "wb") as f:
    pickle.dump(tree_collections, f)

# Summarize trees

with open("tree_summary.txt", "w") as f:
    type_counts_nodes = defaultdict(int)
    type_counts_leaves = defaultdict(int)

    for collection in tree_collections:
        for tree in collection:
            for node in tree.traverse():
                type_counts_nodes[node.x] += 1

                if node.is_leaf():
                    type_counts_leaves[node.x] += 1

    f.write(f"Number of total nodes: {sum(type_counts_nodes.values())}\\n")
    for type in sorted(type_counts_nodes.keys()):
        f.write(f"  Type {type} exists in {type_counts_nodes[type]} nodes\\n")
    f.write("\\n")
    f.write(f"Number of leaves: {sum(type_counts_leaves.values())}\\n")
    for type in sorted(type_counts_leaves.keys()):
        f.write(f"  Type {type} exists in {type_counts_leaves[type]} leaves\\n")

# MCMC config

priors = dict(
    xscale=lambda c: gamma(a=2, scale=1),
    xshift=lambda c: norm(loc=5, scale=1),
    yscale=lambda c: gamma(a=2, scale=1),
    yshift=lambda c: gamma(a=1, scale=1),
    death_rate=lambda c: lognorm(scale=np.exp(0), s=0.3),
    mutation_rate=lambda c: lognorm(scale=np.exp(0), s=0.5),
)

proposals = dict(
    xscale=lambda c: lognorm(scale=c, s=0.3),
    xshift=lambda c: norm(loc=c, scale=1),
    yscale=lambda c: lognorm(scale=c, s=0.3),
    yshift=lambda c: lognorm(scale=c, s=0.3),
    death_rate=lambda c: lognorm(scale=c, s=0.3),
    mutation_rate=lambda c: lognorm(scale=c, s=0.3),
)


def log_likelihood(death_rate, mutation_rate, trees, **birth_params):
    return models.stadler_full_log_likelihood(
        trees=trees,
        birth_response=poisson.SigmoidResponse(**birth_params),
        death_response=poisson.ConstantResponse(death_rate),
        mutation_response=poisson.ConstantResponse(mutation_rate),
        mutator=TRUE_PARAMETERS["mutator"],
        extant_sampling_probability=TRUE_PARAMETERS["extant_sampling_probability"],
        extinct_sampling_probability=TRUE_PARAMETERS["extinct_sampling_probability"],
        present_time=PRESENT_TIME,
    )


# Run MCMC

samples = pd.DataFrame()

for run, trees in enumerate(tree_collections):
    chain = metropolis_hastings(
        length=5000,
        priors=priors,
        proposals=proposals,
        log_likelihood=jit(partial(log_likelihood, trees=trees)),
    )

    chain["run"] = run
    samples = pd.concat((samples, chain))

samples.to_csv("samples.csv")
