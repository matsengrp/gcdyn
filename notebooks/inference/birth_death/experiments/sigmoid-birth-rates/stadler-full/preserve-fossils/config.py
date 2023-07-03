import jax.numpy as np
from scipy.stats import gamma, lognorm, norm

from gcdyn import models, mutators, poisson
from gcdyn.mcmc import Parameter

STATE_SPACE = (2, 4, 6, 8)
INITIAL_STATE = 2
PRESENT_TIME = 1
NUM_TREES = 7
TREE_SEED = 10

TRUE_PARAMETERS = {
    "birth_response": poisson.SigmoidResponse(1.0, 5.0, 3.0, 1.0),
    "death_response": poisson.ConstantResponse(1.3),
    "mutation_response": poisson.ConstantResponse(2),
    "mutator": mutators.DiscreteMutator(
        state_space=STATE_SPACE,
        transition_matrix=np.array(
            [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
        )
        / 3,
    ),
    "extant_sampling_probability": 1,
    "extinct_sampling_probability": 1,
}

XSCALE_PRIOR_SHAPE = 2
XSCALE_PRIOR_SCALE = 1
XSHIFT_PRIOR_MEAN = 5
XSHIFT_PRIOR_SD = 1
YSCALE_PRIOR_SHAPE = 2
YSCALE_PRIOR_SCALE = 1
YSHIFT_PRIOR_SHAPE = 1
YSHIFT_PRIOR_SCALE = 1
DR_PRIOR_MEAN = -1
DR_PRIOR_SD = 0.5

XSCALE_PROPOSAL_SD = 3
XSHIFT_PROPOSAL_SD = 2
YSCALE_PROPOSAL_SD = 1
YSHIFT_PROPOSAL_SD = 1
DR_PROPOSAL_SD = 0.5

MCMC_SEED = 10
NUM_MCMC_SAMPLES = 5000

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
    # yshift=Parameter(
    #     prior_log_density=lambda y: y == TRUE_PARAMETERS["birth_response"].yshift,
    #     prior_generator=lambda n, rng: np.ones(n)
    #     * TRUE_PARAMETERS["birth_response"].yshift,
    #     proposal_log_density=lambda p, c: p == TRUE_PARAMETERS["birth_response"].yshift,
    #     proposal_generator=lambda c, rng: TRUE_PARAMETERS["birth_response"].yshift,
    # ),
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


def log_likelihood(death_rate, trees, **birth_params):
    return models.stadler_full_log_likelihood(
        trees=trees,
        birth_response=poisson.SigmoidResponse(**birth_params),
        death_response=poisson.ConstantResponse(death_rate),
        mutation_response=TRUE_PARAMETERS["mutation_response"],
        mutator=TRUE_PARAMETERS["mutator"],
        extant_sampling_probability=TRUE_PARAMETERS["extant_sampling_probability"],
        extinct_sampling_probability=TRUE_PARAMETERS["extinct_sampling_probability"],
        present_time=PRESENT_TIME,
    )
