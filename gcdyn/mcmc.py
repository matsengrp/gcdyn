import sys

import numpy as np
import pandas as pd

if hasattr(sys.modules["__main__"], "get_ipython"):
    from tqdm import notebook as tqdm
else:
    import tqdm


def metropolis_hastings(
    length, priors, log_likelihood, proposals, initial_value=None, seed=None
):
    rng = np.random.default_rng(seed)

    if not initial_value:
        initial_value = {
            param: priors[param].rvs(1, random_state=rng) for param in priors
        }

    chain = {param: np.zeros(length + 1) for param in initial_value}
    stats = {stat: np.zeros(length + 1) for stat in ("log_likelihood", "log_prior")}

    for param, value in initial_value.items():
        chain[param][0] = value
        stats["log_prior"][0] += priors[param].logpdf(value)

    stats["log_likelihood"][0] = log_likelihood(**initial_value)

    print("Running MH chain...")

    for i in tqdm.trange(length):
        for param in chain:
            # Initialize the next value to be the same as the current value
            current_value = chain[param][i]
            chain[param][i + 1] = current_value

            proposed_value = proposals[param](current_value).rvs(1, random_state=rng)

            # If the proposal distribution is a point mass (ie. we aren't sampling the parameter),
            # don't bother computing the MH ratio
            if proposed_value == current_value:
                continue

            # Calculate the MH ratio. We try to be efficient, reusing previous values where possible
            all_current_values = {param: chain[param][i] for param in chain}
            all_proposed_values = all_current_values | {param: proposed_value}

            try:
                proposed_log_prior = sum(
                    priors[param].logpdf(value)
                    for param, value in all_proposed_values.items()
                )
                proposed_log_likelihood = log_likelihood(**all_proposed_values)

                log_mh_ratio = (
                    proposed_log_prior
                    + proposed_log_likelihood
                    + proposals[param](proposed_value).logpdf(current_value)
                    - stats["log_prior"][i]
                    - stats["log_likelihood"][i]
                    - proposals[param](current_value).logpdf(proposed_value)
                )
            except Exception:
                print("Something went wrong while calculating the MH ratio.")
                print("The current values at time of failure were", all_current_values)
                print("The proposed values at time of failure were", proposals)
                print("The exception raised was:")
                raise

            # Accept or reject the proposal for the next value
            if np.log(rng.uniform()) < log_mh_ratio:
                chain[param][i + 1] = proposed_value

    stats["log_joint"] = stats["log_prior"] + stats["log_likelihood"]

    return pd.DataFrame(dict(**chain, **stats))
