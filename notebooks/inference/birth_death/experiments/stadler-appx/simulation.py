# %%

from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
from jax import jit
from jax.config import config
from scipy.stats import lognorm

from gcdyn import models, mutators, poisson, utils
from gcdyn.mcmc import Parameter, mh_chain

config.update("jax_enable_x64", True)

# %% Configurables

STATE_SPACE = (1, 3)
INITIAL_STATE = 3
PRESENT_TIME = 1
NUM_TREES = 10
TREE_SEED = 20

TRUE_PARAMETERS = {
    "birth_response": poisson.DiscreteResponse(phenotypes=STATE_SPACE, values=(1, 4)),
    "death_response": poisson.ConstantResponse(0.5),
    "mutation_response": poisson.ConstantResponse(0.2),
    "mutator": mutators.DiscreteMutator(
        state_space=STATE_SPACE,
        transition_matrix=np.array([[0, 1], [1, 0]]),
    ),
    "extant_sampling_probability": 1,
    "extinct_sampling_probability": 0,
}

BR1_PRIOR_MEAN = 1.5
BR1_PRIOR_SD = 1
BR2_PRIOR_MEAN = 1.5
BR2_PRIOR_SD = 1

BR1_PROPOSAL_SD = 0.5
BR2_PROPOSAL_SD = 0.5

MCMC_PARAMETERS = dict(
    # Note that lognorm(a, b) in R is lognorm(scale=exp(a), s=b) in scipy
    birth_rate1=Parameter(
        prior_log_density=lognorm(scale=np.exp(BR1_PRIOR_MEAN), s=BR1_PRIOR_SD).logpdf,
        prior_generator=lambda n: lognorm(
            scale=np.exp(BR1_PRIOR_MEAN), s=BR1_PRIOR_SD
        ).rvs(n),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=BR1_PROPOSAL_SD).logpdf(p),
        proposal_generator=lambda c: lognorm(scale=c, s=BR1_PROPOSAL_SD).rvs(1),
    ),
    birth_rate2=Parameter(
        prior_log_density=lognorm(scale=np.exp(BR2_PRIOR_MEAN), s=BR2_PRIOR_SD).logpdf,
        prior_generator=lambda n: lognorm(
            scale=np.exp(BR2_PRIOR_MEAN), s=BR2_PRIOR_SD
        ).rvs(n),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=BR2_PROPOSAL_SD).logpdf(p),
        proposal_generator=lambda c: lognorm(scale=c, s=BR2_PROPOSAL_SD).rvs(1),
    ),
)


# %% Sample trees

trees = utils.sample_trees(
    n=NUM_TREES,
    t=PRESENT_TIME,
    init_x=INITIAL_STATE,
    **TRUE_PARAMETERS,
    seed=TREE_SEED,
)

for tree in trees:
    if TRUE_PARAMETERS["extinct_sampling_probability"] == 1:
        tree._pruned = True
    elif TRUE_PARAMETERS["extinct_sampling_probability"] == 0:
        tree.prune()

print(
    "After pruning: average of",
    sum(len(list(tree.traverse())) for tree in trees) / len(trees),
    "nodes per tree, and average of",
    sum(len(list(tree.iter_leaves())) for tree in trees) / len(trees),
    "leaves per tree, over",
    len(trees),
    "trees.",
    "\n",
)

type_counts = defaultdict(int)

for tree in trees:
    for node in tree.traverse():
        type_counts[node.x] += 1

for type in sorted(type_counts.keys()):
    print(f"Type {type} exists in {type_counts[type]} nodes")

# %% Configure likelihoods

log_likelihood_bases = []
log_likelihoods = []

for tree in trees:
    ll_base = jit(
        partial(
            models.stadler_appx_log_likelihood,
            trees=[tree],
            death_response=TRUE_PARAMETERS["death_response"],
            mutation_response=TRUE_PARAMETERS["mutation_response"],
            mutator=TRUE_PARAMETERS["mutator"],
            extant_sampling_probability=TRUE_PARAMETERS["extant_sampling_probability"],
            extinct_sampling_probability=TRUE_PARAMETERS[
                "extinct_sampling_probability"
            ],
            present_time=PRESENT_TIME,
        )
    )

    log_likelihood_bases.append(ll_base)

    def ll(birth_rate1, birth_rate2):
        return ll_base(
            birth_response=poisson.DiscreteResponse(
                phenotypes=STATE_SPACE,
                values=np.hstack((birth_rate1, birth_rate2)),
            ),
        )

    log_likelihoods.append(ll)


# %% Run MCMC


def mh(ll):
    return mh_chain(
        length=2000,
        parameters=MCMC_PARAMETERS,
        log_likelihood=ll,
    )


# Can't use multiprocessing with JAX...
results = map(mh, log_likelihoods)

dataframes = []

for i, result in enumerate(results):
    # result == [posterior_samples, stats]
    df = dict(tree=i + 1, **result[0], **result[1])
    dataframes.append(pd.DataFrame(df))

pd.concat(dataframes).to_csv("samples.csv")

# %%
