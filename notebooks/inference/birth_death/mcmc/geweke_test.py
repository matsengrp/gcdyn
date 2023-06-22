# %%
import pickle
from functools import partial

import numpy as np
import pandas as pd
import tqdm
from jax import disable_jit
from jax.config import config
from mcmc import mh_step
from scipy.stats import gamma, lognorm

from gcdyn import models, mutators, poisson, utils, bdms

config.update("jax_enable_x64", True)

initial_parameter_values = {
    "birth_response": poisson.ConstantResponse(2),
    "death_response": poisson.ConstantResponse(1),
}

true_parameters = {
    "mutation_response": poisson.ConstantResponse(0),
    "mutator": mutators.DiscreteMutator(
        state_space=(1, 2, 3),
        transition_matrix=utils.random_transition_matrix(length=3),
    ),
    "extant_sampling_probability": 1,
}

NUM_SAMPLES = 2000
PRESENT_TIME = 2


BIRTH_PROPOSAL_SD = 0.2
DEATH_PROPOSAL_SD = 0.2

BIRTH_PRIOR_SHAPE = 2
BIRTH_PRIOR_RATE = 3

DEATH_PRIOR_SHAPE = 3.5
DEATH_PRIOR_RATE = 3


def sample_prior():
    return {
        "birth_response": poisson.ConstantResponse(
            gamma(a=BIRTH_PRIOR_SHAPE, scale=1 / BIRTH_PRIOR_RATE).rvs(size=1).item()
        ),
        "death_response": poisson.ConstantResponse(
            gamma(a=BIRTH_PRIOR_SHAPE, scale=1 / DEATH_PRIOR_RATE).rvs(size=1).item()
        ),
    }


prior_log_densities = {
    "birth_response": lambda response: gamma(
        a=BIRTH_PRIOR_SHAPE, scale=1 / BIRTH_PRIOR_RATE
    ).logpdf(response.value),
    "death_response": lambda response: gamma(
        a=BIRTH_PRIOR_SHAPE, scale=1 / DEATH_PRIOR_RATE
    ).logpdf(response.value),
}

proposal_generators = {
    "birth_response": lambda current: poisson.ConstantResponse(
        lognorm(scale=current.value, s=BIRTH_PROPOSAL_SD).rvs(size=1).item()
    ),
    "death_response": lambda current: poisson.ConstantResponse(
        lognorm(scale=current.value, s=DEATH_PROPOSAL_SD).rvs(size=1).item()
    ),
}

proposal_log_densities = {
    "birth_response": lambda p, c: lognorm(scale=c.value, s=BIRTH_PROPOSAL_SD).logpdf(
        p.value
    ),
    "death_response": lambda p, c: lognorm(scale=c.value, s=DEATH_PROPOSAL_SD).logpdf(
        p.value
    ),
}

# Function of just birth_rate and death_rate
sample_tree = partial(
    bdms.sample_trees,
    n=1,
    t=PRESENT_TIME,
    mutation_response=true_parameters["mutation_response"],
    mutator=true_parameters["mutator"],
    extant_sampling_probability=true_parameters["extant_sampling_probability"],
    min_survivors=0,
)

log_likelihood = partial(
    models.stadler_full_log_likelihood,
    mutation_response=true_parameters["mutation_response"],
    mutator=true_parameters["mutator"],
    extant_sampling_probability=true_parameters["extant_sampling_probability"],
    extinct_sampling_probability=1,
    present_time=PRESENT_TIME,
)

# %%
# Monte Carlo sampling
print("Running MC sampling...")

mc_samples = [sample_prior() for _ in range(NUM_SAMPLES)]

mc_samples = {
    "birth_response": np.hstack([s["birth_response"].value for s in mc_samples]),
    "death_response": np.hstack([s["death_response"].value for s in mc_samples]),
}

# %%
# Gibbs sampling
print("Running MCMC sampling...")

mcmc_samples = [initial_parameter_values]

with disable_jit():
    for iteration in tqdm.trange(NUM_SAMPLES):
        trees = sample_tree(**mcmc_samples[-1], print_info=False)

        for tree in trees:
            tree._pruned = True

        params = mh_step(
            from_values=mcmc_samples[-1],
            log_priors=prior_log_densities,
            log_likelihood=partial(log_likelihood, trees=trees),
            proposal_generators=proposal_generators,
            proposal_log_densities=proposal_log_densities,
        )

        mcmc_samples.append(params)

mcmc_samples = {
    "birth_response": np.hstack([s["birth_response"].value for s in mcmc_samples]),
    "death_response": np.hstack([s["death_response"].value for s in mcmc_samples]),
}

# %%
print("Exporting samples...")

with open("geweke_samples.pkl", "wb") as f:
    pickle.dump({"mc_samples": mc_samples, "mcmc_samples": mcmc_samples}, f)

mc_samples = pd.DataFrame(mc_samples).assign(Method="MC")
mc_samples["t"] = mc_samples.index + 1
mcmc_samples = pd.DataFrame(mcmc_samples).assign(Method="MCMC")
mcmc_samples["t"] = mcmc_samples.index + 1
pd.concat([mc_samples, mcmc_samples]).to_csv("geweke_samples.csv")

print("Done.")
