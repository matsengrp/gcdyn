import jax.numpy as np
from scipy.stats import lognorm
from tree_config import PRESENT_TIME, STATE_SPACE, TRUE_PARAMETERS

from gcdyn import models, poisson
from gcdyn.mcmc import Parameter

BR_PRIOR_MEAN = 1.5
BR_PRIOR_SD = 1
DR_PRIOR_MEAN = 0  # -1
DR_PRIOR_SD = 0.3  # 0.5

BR_PROPOSAL_SD = 0.1
DR_PROPOSAL_SD = 0.2

MCMC_SEED = 10
NUM_MCMC_SAMPLES = 5000

MCMC_PARAMETERS = dict(
    # Note that lognorm(a, b) in R is lognorm(scale=exp(a), s=b) in scipy
    birth_rate1=Parameter(
        prior_log_density=lognorm(scale=np.exp(BR_PRIOR_MEAN), s=BR_PRIOR_SD).logpdf,
        prior_generator=lambda n, rng: lognorm(
            scale=np.exp(BR_PRIOR_MEAN), s=BR_PRIOR_SD
        ).rvs(n, random_state=rng),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=BR_PROPOSAL_SD).logpdf(p),
        proposal_generator=lambda c, rng: lognorm(scale=c, s=BR_PROPOSAL_SD).rvs(
            1, random_state=rng
        ),
    ),
    birth_rate2=Parameter(
        prior_log_density=lognorm(scale=np.exp(BR_PRIOR_MEAN), s=BR_PRIOR_SD).logpdf,
        prior_generator=lambda n, rng: lognorm(
            scale=np.exp(BR_PRIOR_MEAN), s=BR_PRIOR_SD
        ).rvs(n, random_state=rng),
        proposal_log_density=lambda p, c: lognorm(scale=c, s=BR_PROPOSAL_SD).logpdf(p),
        proposal_generator=lambda c, rng: lognorm(scale=c, s=BR_PROPOSAL_SD).rvs(
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
)


def log_likelihood(birth_rate1, birth_rate2, death_rate, trees):
    return models.stadler_appx_log_likelihood(
        trees=trees,
        birth_response=poisson.DiscreteResponse(
            phenotypes=STATE_SPACE,
            values=np.hstack(
                (
                    birth_rate1,
                    birth_rate2,
                )
            ),
        ),
        death_response=poisson.ConstantResponse(death_rate),
        mutation_response=TRUE_PARAMETERS["mutation_response"],
        mutator=TRUE_PARAMETERS["mutator"],
        extant_sampling_probability=TRUE_PARAMETERS["extant_sampling_probability"],
        extinct_sampling_probability=TRUE_PARAMETERS["extinct_sampling_probability"],
        present_time=PRESENT_TIME,
    )
