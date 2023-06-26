import jax.numpy as np
from scipy.stats import lognorm

from gcdyn import models, mutators, poisson
from gcdyn.mcmc import Parameter

STATE_SPACE = (2, 4, 6, 8)
INITIAL_STATE = 2
PRESENT_TIME = 2
NUM_TREES = 10
TREE_SEED = 11

TRUE_PARAMETERS = {
    "birth_response": poisson.DiscreteResponse(
        phenotypes=STATE_SPACE, values=(1, 2, 3, 4)
    ),
    "death_response": poisson.ConstantResponse(0.5),
    "mutation_response": poisson.ConstantResponse(0.5),
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

BR_PRIOR_MEAN = 1.5
BR_PRIOR_SD = 1
DR_PRIOR_MEAN = -1
DR_PRIOR_SD = 0.5

BR1_PROPOSAL_SD = 0.5
BR2_PROPOSAL_SD = 0.5
DR_PROPOSAL_SD = 0.5

MCMC_PARAMETERS = dict(
    # Note that lognorm(a, b) in R is lognorm(scale=exp(a), s=b) in scipy
    birth_rate1=Parameter(
        prior_log_density=lognorm(scale=np.exp(BR_PRIOR_MEAN), s=BR_PRIOR_SD).logpdf,
        prior_generator=lambda n: lognorm(
            scale=np.exp(BR_PRIOR_MEAN), s=BR_PRIOR_SD
        ).rvs(n),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=BR1_PROPOSAL_SD).logpdf(p),
        proposal_generator=lambda c: lognorm(scale=c, s=BR1_PROPOSAL_SD).rvs(1),
    ),
    birth_rate2=Parameter(
        prior_log_density=lognorm(scale=np.exp(BR_PRIOR_MEAN), s=BR_PRIOR_SD).logpdf,
        prior_generator=lambda n: lognorm(
            scale=np.exp(BR_PRIOR_MEAN), s=BR_PRIOR_SD
        ).rvs(n),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=BR2_PROPOSAL_SD).logpdf(p),
        proposal_generator=lambda c: lognorm(scale=c, s=BR2_PROPOSAL_SD).rvs(1),
    ),
    birth_rate3=Parameter(
        prior_log_density=lognorm(scale=np.exp(BR_PRIOR_MEAN), s=BR_PRIOR_SD).logpdf,
        prior_generator=lambda n: lognorm(
            scale=np.exp(BR_PRIOR_MEAN), s=BR_PRIOR_SD
        ).rvs(n),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=BR2_PROPOSAL_SD).logpdf(p),
        proposal_generator=lambda c: lognorm(scale=c, s=BR2_PROPOSAL_SD).rvs(1),
    ),
    birth_rate4=Parameter(
        prior_log_density=lognorm(scale=np.exp(BR_PRIOR_MEAN), s=BR_PRIOR_SD).logpdf,
        prior_generator=lambda n: lognorm(
            scale=np.exp(BR_PRIOR_MEAN), s=BR_PRIOR_SD
        ).rvs(n),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=BR2_PROPOSAL_SD).logpdf(p),
        proposal_generator=lambda c: lognorm(scale=c, s=BR2_PROPOSAL_SD).rvs(1),
    ),
    death_rate=Parameter(
        prior_log_density=lognorm(scale=np.exp(DR_PRIOR_MEAN), s=DR_PRIOR_SD).logpdf,
        prior_generator=lambda n: lognorm(
            scale=np.exp(DR_PRIOR_MEAN), s=DR_PRIOR_SD
        ).rvs(n),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=DR_PROPOSAL_SD).logpdf(p),
        proposal_generator=lambda c: lognorm(scale=c, s=DR_PROPOSAL_SD).rvs(1),
    ),
)


def log_likelihood(
    birth_rate1, birth_rate2, birth_rate3, birth_rate4, death_rate, trees
):
    return models.stadler_appx_log_likelihood(
        trees=trees,
        birth_response=poisson.DiscreteResponse(
            phenotypes=STATE_SPACE,
            values=np.hstack((birth_rate1, birth_rate2, birth_rate3, birth_rate4)),
        ),
        death_response=poisson.ConstantResponse(death_rate),
        mutation_response=TRUE_PARAMETERS["mutation_response"],
        mutator=TRUE_PARAMETERS["mutator"],
        extant_sampling_probability=TRUE_PARAMETERS["extant_sampling_probability"],
        extinct_sampling_probability=TRUE_PARAMETERS["extinct_sampling_probability"],
        present_time=PRESENT_TIME,
    )
