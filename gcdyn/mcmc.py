import sys

import numpy as np
import numpy.random as random

if hasattr(sys.modules["__main__"], "get_ipython"):
    from tqdm import notebook as tqdm
else:
    import tqdm


class Parameter:
    def __init__(
        self,
        prior_log_density,
        prior_generator,
        proposal_log_density,
        proposal_generator,
    ):
        """
        prior_log_density: function(parameter value)
        prior_generator: function(num samples)
        proposal_log_density: function(proposed value, current value)
            If None, assumed to be a symmetric proposal
        proposal_generator: function(current value)
        """
        self.prior_log_density = prior_log_density
        self.prior_generator = prior_generator
        self.proposal_log_density = proposal_log_density
        self.proposal_generator = proposal_generator

        if not self.proposal_log_density:
            self.proposal_log_density = lambda p, c: 0


def mh_step(
    from_values,
    parameters,
    log_likelihood,
):
    """
    from_values (dict): parameter key => current parameter value
    parameter_config (dict): parameter key => `Parameter` object
    log_likelihood (callable): function with named args for every parameter key
    """

    assert from_values.keys() == parameters.keys()

    for name, param in parameters.items():
        log_prior = param.prior_log_density
        proposal_log_density = param.proposal_log_density
        proposal_generator = param.proposal_generator

        proposals = from_values | {name: proposal_generator(from_values[name])}

        try:
            log_mh_ratio = (
                log_prior(proposals[name])
                + log_likelihood(**proposals)
                + proposal_log_density(from_values[name], proposals[name])
                - log_prior(from_values[name])
                - log_likelihood(**from_values)
                - proposal_log_density(proposals[name], from_values[name])
            )
        except Exception:
            print("Something went wrong while calculating the MH ratio.")
            print("The current values at time of failure were", from_values)
            print("The proposed values at time of failure were", proposals)
            print("The exception raised was:")
            raise

        rng = random.default_rng()

        if rng.uniform() < np.minimum(1, np.exp(log_mh_ratio)):
            from_values = proposals

    return from_values


def mh_chain(length, parameters, log_likelihood, initial_value=None):
    if not initial_value:
        initial_value = {
            name: param.prior_generator(1) for name, param in parameters.items()
        }

    chain = [initial_value]

    print("Running MH chain...")

    for _ in tqdm.trange(length):
        sample = mh_step(
            from_values=chain[-1],
            parameters=parameters,
            log_likelihood=log_likelihood,
        )
        chain.append(sample)

    stats = {
        "log_likelihood": np.hstack([log_likelihood(**sample) for sample in chain]),
    }

    chain = {
        param: np.hstack([sample[param] for sample in chain]) for param in initial_value
    }

    stats["log_prior"] = sum(
        parameters[name].prior_log_density(samples) for name, samples in chain.items()
    )
    stats["log_joint"] = stats["log_prior"] + stats["log_likelihood"]

    return chain, stats


if __name__ == "__main__":
    # Run a sanity check

    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import beta, binom, norm

    parameter_config = {
        "p": Parameter(
            prior_log_density=beta(1, 1).logpdf,
            prior_generator=lambda n: beta(1, 1).rvs(n),
            proposal_log_density=lambda p, c: norm(loc=c).logpdf(p),
            proposal_generator=lambda c: norm(loc=c).rvs(size=1),
        )
    }

    samples = mh_chain(
        length=3000,
        initial_value={"p": 0},
        log_likelihood=lambda p: binom(5, p).logpmf(3),
        parameters=parameter_config,
    )

    true_samples = beta(1 + 3, 1 + 2).rvs(size=100000)

    fig, ax = plt.subplots()
    sns.histplot(samples["p"], stat="density", ax=ax)
    sns.kdeplot(true_samples, ax=ax)
