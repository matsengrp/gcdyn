import pickle
from collections import defaultdict
from functools import partial

import jax.numpy as np
import pandas as pd
from jax import jit
from jax.config import config
from scipy.stats import gamma, lognorm, norm

from gcdyn import models, mutators, poisson
from gcdyn.mcmc import Parameter, mh_chain
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

XSCALE_PRIOR_SHAPE = 2
XSCALE_PRIOR_SCALE = 1
XSHIFT_PRIOR_MEAN = 5
XSHIFT_PRIOR_SD = 1
YSCALE_PRIOR_SHAPE = 2
YSCALE_PRIOR_SCALE = 1
YSHIFT_PRIOR_SHAPE = 1
YSHIFT_PRIOR_SCALE = 1
DR_PRIOR_MEAN = 0
DR_PRIOR_SD = 0.3
MR_PRIOR_MEAN = 0
MR_PRIOR_SD = 0.5

XSCALE_PROPOSAL_SD = 0.3
XSHIFT_PROPOSAL_SD = 1
YSCALE_PROPOSAL_SD = 0.3
YSHIFT_PROPOSAL_SD = 0.3
DR_PROPOSAL_SD = 0.3
MR_PROPOSAL_SD = 0.3

MCMC_PARAMETERS = dict(
    xscale=Parameter(
        prior_log_density=gamma(a=XSCALE_PRIOR_SHAPE, scale=XSCALE_PRIOR_SCALE).logpdf,
        prior_generator=lambda n, rng: gamma(
            a=XSCALE_PRIOR_SHAPE, scale=XSCALE_PRIOR_SCALE
        ).rvs(n, random_state=rng),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=XSCALE_PROPOSAL_SD).logpdf(
            p
        ),
        proposal_generator=lambda c, rng: lognorm(scale=c, s=XSCALE_PROPOSAL_SD).rvs(
            1, random_state=rng
        ),
    ),
    xshift=Parameter(
        prior_log_density=norm(loc=XSHIFT_PRIOR_MEAN, scale=XSHIFT_PRIOR_SD).logpdf,
        prior_generator=lambda n, rng: norm(
            loc=XSHIFT_PRIOR_MEAN, scale=XSHIFT_PRIOR_SD
        ).rvs(n, random_state=rng),
        proposal_log_density=lambda p, c: norm(loc=c, scale=XSHIFT_PROPOSAL_SD).logpdf(
            p
        ),
        proposal_generator=lambda c, rng: norm(loc=c, scale=XSHIFT_PROPOSAL_SD).rvs(
            1, random_state=rng
        ),
    ),
    yscale=Parameter(
        prior_log_density=gamma(a=YSCALE_PRIOR_SHAPE, scale=YSCALE_PRIOR_SCALE).logpdf,
        prior_generator=lambda n, rng: gamma(
            a=YSCALE_PRIOR_SHAPE, scale=YSCALE_PRIOR_SCALE
        ).rvs(n, random_state=rng),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=YSCALE_PROPOSAL_SD).logpdf(
            p
        ),
        proposal_generator=lambda c, rng: lognorm(scale=c, s=YSCALE_PROPOSAL_SD).rvs(
            1, random_state=rng
        ),
    ),
    yshift=Parameter(
        prior_log_density=gamma(a=YSHIFT_PRIOR_SHAPE, scale=YSHIFT_PRIOR_SCALE).logpdf,
        prior_generator=lambda n, rng: gamma(
            a=YSHIFT_PRIOR_SHAPE, scale=YSHIFT_PRIOR_SCALE
        ).rvs(n, random_state=rng),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=YSHIFT_PROPOSAL_SD).logpdf(
            p
        ),
        proposal_generator=lambda c, rng: lognorm(scale=c, s=YSHIFT_PROPOSAL_SD).rvs(
            1, random_state=rng
        ),
    ),
    death_rate=Parameter(
        prior_log_density=lognorm(scale=np.exp(DR_PRIOR_MEAN), s=DR_PRIOR_SD).logpdf,
        prior_generator=lambda n, rng: lognorm(
            scale=np.exp(DR_PRIOR_MEAN), s=DR_PRIOR_SD
        ).rvs(n, random_state=rng),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=DR_PROPOSAL_SD).logpdf(p),
        proposal_generator=lambda c, rng: lognorm(scale=c, s=DR_PROPOSAL_SD).rvs(
            1, random_state=rng
        ),
    ),
    mutation_rate=Parameter(
        prior_log_density=lognorm(scale=np.exp(MR_PRIOR_MEAN), s=MR_PRIOR_SD).logpdf,
        prior_generator=lambda n, rng: lognorm(
            scale=np.exp(MR_PRIOR_MEAN), s=MR_PRIOR_SD
        ).rvs(n, random_state=rng),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=MR_PROPOSAL_SD).logpdf(p),
        proposal_generator=lambda c, rng: lognorm(scale=c, s=MR_PROPOSAL_SD).rvs(
            1, random_state=rng
        ),
    ),
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

with open("trees.pkl", "rb") as f:
    tree_collections = pickle.load(f)

samples = pd.DataFrame()

for run, trees in enumerate(tree_collections):
    posterior_samples, stats = mh_chain(
        length=5000,
        parameters=MCMC_PARAMETERS,
        log_likelihood=jit(partial(log_likelihood, trees=trees)),
    )

    samples = pd.concat(
        (samples, pd.DataFrame(dict(**posterior_samples, **stats, run=run)))
    )

samples.to_csv("samples.csv")
