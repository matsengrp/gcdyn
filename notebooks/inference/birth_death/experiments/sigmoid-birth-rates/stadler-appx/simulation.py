# %%

from functools import partial

import jax.numpy as np
import pandas as pd
from jax import jit
from jax.config import config
from scipy.stats import gamma, lognorm, norm

from gcdyn import models, mutators, poisson, utils
from gcdyn.mcmc import Parameter, mh_chain

config.update("jax_enable_x64", True)

# %% Configurables

STATE_SPACE = (1, 3)
INITIAL_STATE = 3
PRESENT_TIME = 5
NUM_TREES = 5
TREE_SEED = 20

TRUE_PARAMETERS = {
    "birth_response": poisson.SigmoidResponse(1.0, 5.0, 2.0, 0.5),
    "death_response": poisson.ConstantResponse(0.5),
    "mutation_response": poisson.ConstantResponse(0.2),
    "mutator": mutators.DiscreteMutator(
        state_space=STATE_SPACE,
        transition_matrix=np.array([[0, 1], [1, 0]]),
    ),
    "extant_sampling_probability": 1,
    "extinct_sampling_probability": 0,
}

XSCALE_PRIOR_SHAPE = 2
XSCALE_PRIOR_SCALE = 1
XSHIFT_PRIOR_MEAN = 5
XSHIFT_PRIOR_SD = 1
YSCALE_PRIOR_SHAPE = 2
YSCALE_PRIOR_SCALE = 1
DR_PRIOR_SHAPE = 3.5
DR_PRIOR_SCALE = 1 / 3

XSCALE_PROPOSAL_SD = 3
XSHIFT_PROPOSAL_SD = 2
YSCALE_PROPOSAL_SD = 1
DR_PROPOSAL_SD = 1 / 2

MCMC_PARAMETERS = dict(
    xscale=Parameter(
        prior_log_density=gamma(a=XSCALE_PRIOR_SHAPE, scale=XSCALE_PRIOR_SCALE).logpdf,
        prior_generator=lambda n: gamma(
            a=XSCALE_PRIOR_SHAPE, scale=XSCALE_PRIOR_SCALE
        ).rvs(n),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=XSCALE_PROPOSAL_SD).logpdf(
            p
        ),
        proposal_generator=lambda c: lognorm(scale=c, s=XSCALE_PROPOSAL_SD).rvs(1),
    ),
    xshift=Parameter(
        prior_log_density=norm(loc=XSHIFT_PRIOR_MEAN, scale=XSHIFT_PRIOR_SD).logpdf,
        prior_generator=lambda n: norm(
            loc=XSHIFT_PRIOR_MEAN, scale=XSHIFT_PRIOR_SD
        ).rvs(n),
        proposal_log_density=lambda p, c: norm(loc=c, scale=XSHIFT_PROPOSAL_SD).logpdf(
            p
        ),
        proposal_generator=lambda c: norm(loc=c, scale=XSHIFT_PROPOSAL_SD).rvs(1),
    ),
    yscale=Parameter(
        prior_log_density=gamma(a=YSCALE_PRIOR_SHAPE, scale=YSCALE_PRIOR_SCALE).logpdf,
        prior_generator=lambda n: gamma(
            a=YSCALE_PRIOR_SHAPE, scale=YSCALE_PRIOR_SCALE
        ).rvs(n),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=YSCALE_PROPOSAL_SD).logpdf(
            p
        ),
        proposal_generator=lambda c: lognorm(scale=c, s=YSCALE_PROPOSAL_SD).rvs(1),
    ),
    yshift=Parameter(
        prior_log_density=lambda y: y == TRUE_PARAMETERS["birth_response"].yshift,
        prior_generator=lambda n: np.ones(n) * TRUE_PARAMETERS["birth_response"].yshift,
        proposal_log_density=lambda p, c: p == TRUE_PARAMETERS["birth_response"].yshift,
        proposal_generator=lambda c: TRUE_PARAMETERS["birth_response"].yshift,
    ),
    death_rate=Parameter(
        prior_log_density=gamma(a=DR_PRIOR_SHAPE, scale=DR_PRIOR_SCALE).logpdf,
        prior_generator=lambda n: gamma(a=DR_PRIOR_SHAPE, scale=DR_PRIOR_SCALE).rvs(n),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=DR_PROPOSAL_SD).logpdf(p),
        proposal_generator=lambda c: lognorm(scale=c, s=DR_PROPOSAL_SD).rvs(1),
    ),
)


def log_likelihood(death_rate, trees, **birth_params):
    return models.stadler_appx_log_likelihood(
        trees=trees,
        birth_response=poisson.SigmoidResponse(**birth_params),
        death_response=poisson.ConstantResponse(death_rate),
        mutation_response=TRUE_PARAMETERS["mutation_response"],
        mutator=TRUE_PARAMETERS["mutator"],
        extant_sampling_probability=TRUE_PARAMETERS["extant_sampling_probability"],
        extinct_sampling_probability=TRUE_PARAMETERS["extinct_sampling_probability"],
        present_time=PRESENT_TIME,
    )


# %% Run MCMC

trees = utils.sample_trees(
    n=NUM_TREES,
    t=PRESENT_TIME,
    init_x=INITIAL_STATE,
    **TRUE_PARAMETERS,
    seed=TREE_SEED,
    min_survivors=0,
    prune=True,
)


def mh(tree):
    return mh_chain(
        length=2000,
        parameters=MCMC_PARAMETERS,
        log_likelihood=jit(partial(log_likelihood, trees=[tree])),
    )


results = map(mh, trees)

dataframes = []

for i, result in enumerate(results):
    # result == [posterior_samples, stats]
    df = dict(tree=i + 1, **result[0], **result[1])
    dataframes.append(pd.DataFrame(df))

pd.concat(dataframes).to_csv("samples.csv")

# %%
