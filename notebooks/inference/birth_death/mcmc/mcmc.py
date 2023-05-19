from functools import partial
import numpy as np
import numpy.random as random
import sys

if hasattr(sys.modules["__main__"], "get_ipython"):
    from tqdm import notebook as tqdm
else:
    import tqdm


def mh_step(
    from_values,
    log_priors,
    log_likelihood,
    proposal_generators,
    proposal_log_densities=None,
):
    """
    from_values (dict): parameter key => current parameter value
    log_priors (dict): parameter key => function(parameter value)
    log_likelihood (callable): function with named args for every parameter key
    proposal_generator (dict): parameter key => function(proposed value, current value)
    proposal_log_densities (dict): parameter key => function(parameter value)
        If None (or => None), assumed to be symmetric proposal
    """

    if not proposal_log_densities:
        proposal_log_densities = {key: lambda p, c: 0 for key in from_values}
    else:
        for key, density in proposal_log_densities.items():
            if not density:
                proposal_log_densities[key] = lambda p, c: 0

    assert (
        from_values.keys()
        == log_priors.keys()
        == proposal_generators.keys()
        == proposal_log_densities.keys()
    )

    for key in list(from_values):
        proposals = from_values | {key: proposal_generators[key](from_values[key])}

        try:
            log_mh_ratio = (
                log_priors[key](proposals[key])
                + log_likelihood(**proposals)
                + proposal_log_densities[key](from_values[key], proposals[key])
                - log_priors[key](from_values[key])
                - log_likelihood(**from_values)
                - proposal_log_densities[key](proposals[key], from_values[key])
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

    diagnostics = dict()
    diagnostics["log_prior"] = sum(
        log_priors[key](from_values[key]) for key in from_values
    )
    diagnostics["log_likelihood"] = log_likelihood(**from_values)
    diagnostics["log_posterior"] = (
        diagnostics["log_prior"] + diagnostics["log_likelihood"]
    )

    return from_values, diagnostics


def mh_tour(num_samples, initial_value, **step_kwargs):
    samples = [initial_value]
    diagnostics = []

    step = partial(mh_step, **step_kwargs)

    for _ in tqdm.trange(num_samples):
        sample, diag = step(from_values=samples[-1])
        samples.append(sample)
        diagnostics.append(diag)

    samples = {
        key: np.hstack([sample[key] for sample in samples]) for key in initial_value
    }

    diagnostics = {
        key: np.hstack([diag[key] for diag in diagnostics]) for key in diagnostics[0]
    }

    return samples, diagnostics


if __name__ == "__main__":
    # Run a sanity check

    from scipy.stats import beta, binom, norm
    import matplotlib.pyplot as plt
    import seaborn as sns

    samples = mh_tour(
        num_samples=3000,
        initial_value={"p": 0},
        log_priors={"p": lambda p: beta(1, 1).logpdf(p)},
        log_likelihood=lambda p: binom(5, p).logpmf(3),
        proposal_generators={"p": lambda c: norm(loc=c).rvs(size=1)},
        proposal_log_densities={"p": lambda p, c: norm(loc=c).logpdf(p)},
    )

    true_samples = beta(1 + 3, 1 + 2).rvs(size=100000)

    fig, ax = plt.subplots()
    sns.histplot(samples["p"], stat="density", ax=ax)
    sns.kdeplot(true_samples, ax=ax)
