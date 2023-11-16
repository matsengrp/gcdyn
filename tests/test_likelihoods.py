import unittest

from numpy import exp, log, max, sqrt
from scipy.stats import expon

from gcdyn import bdms, models, mutators, poisson, utils

BIRTH = bdms.TreeNode._BIRTH_EVENT
DEATH = bdms.TreeNode._DEATH_EVENT
MUTATION = bdms.TreeNode._MUTATION_EVENT
SAMPLED_SURVIVOR = bdms.TreeNode._SAMPLING_EVENT
UNSAMPLED_SURVIVOR = bdms.TreeNode._SURVIVAL_EVENT

# Number of decimal places to which log-likelihoods should be accurate
TOLERANCE = 2


def add_event(node, event, edge_length):
    child = bdms.TreeNode(t=node.t + edge_length, dist=edge_length)
    child.x = node.x

    child.event = event
    node.add_child(child)

    return child


def log_dexp(x, rate):
    return expon.logpdf(x, scale=1 / rate)


def log_hazard_exp(x, rate):
    return expon.logsf(x, scale=1 / rate)


class TestLikelihoods(unittest.TestCase):
    """Test cases where various MTBD process likelihoods should match."""

    def setUp(self):
        self.λ = poisson.SigmoidResponse(1.0, 0.0, 2.0, 0.0)
        self.μ = poisson.ConstantResponse(1.3)
        self.γ = poisson.ConstantResponse(1.2)
        self.state_space = (1, 1.5, 2)
        self.mutator = mutators.DiscreteMutator(
            state_space=self.state_space,
            transition_matrix=utils.random_transition_matrix(length=3, seed=10),
        )
        self.ρ = 1
        self.σ = 1
        self.Λ = lambda x: self.λ(x) + self.μ(x) + self.γ(x)

        self.parameter_dict = {
            "birth_response": self.λ,
            "death_response": self.μ,
            "mutation_response": self.γ,
            "mutator": self.mutator,
            "extant_sampling_probability": self.ρ,
        }

    def compare_models(self, tree):
        """
        Compares the likelihoods of the naive, Stadler approximate, and Stadler full models,
        and returns the value should they match.

        `tree` should be unpruned, which is a requirement of the naive model.
        """

        # Naive likelihood by code
        naive_ll_code = models.naive_log_likelihood(
            [tree],
            **self.parameter_dict,
        ).item()

        # Stadler approximate likelihood by code
        tree._pruned = True

        present_time = max([node.t for node in tree.iter_leaves()])

        appx_ll_code = models.stadler_appx_log_likelihood(
            [tree],
            **self.parameter_dict,
            extinct_sampling_probability=self.σ,
            present_time=present_time,
        ).item()

        # Stadler full likelihood by code
        full_ll_code = models.stadler_full_log_likelihood(
            [tree],
            **self.parameter_dict,
            extinct_sampling_probability=self.σ,
            present_time=present_time,
        ).item()

        self.assertAlmostEqual(naive_ll_code, appx_ll_code, places=TOLERANCE)
        self.assertAlmostEqual(naive_ll_code, full_ll_code, places=TOLERANCE)

        return naive_ll_code

    def test_sample_event(self):
        """Single edge to sample time."""

        tree = bdms.TreeNode()
        tree.x = self.state_space[0]
        event = add_event(tree, SAMPLED_SURVIVOR, edge_length=3)
        tree._sampled = True  # We do this ourselves

        ll_code = self.compare_models(tree)

        # Naive likelihood by hand.
        # The likelihood is the probability that no BDM event happens along this branch (ie. one would've happened after sample time),
        # times the probability of being sampled
        naive_ll_hand = log_hazard_exp(event.dist, self.Λ(event.up)) + log(self.ρ)

        # Stadler approximate likelihood by hand
        present_time = max([node.t for node in tree.get_leaves()])
        appx_ll_hand = self.log_f_N(event, present_time) + log(self.ρ)

        self.assertAlmostEqual(ll_code, naive_ll_hand, places=TOLERANCE)
        self.assertAlmostEqual(ll_code, appx_ll_hand, places=TOLERANCE)

    def test_death_event(self):
        """Single edge dying and being sampled."""

        tree = bdms.TreeNode()
        tree.x = self.state_space[0]
        event = add_event(tree, DEATH, edge_length=3)
        tree._sampled = True  # We do this ourselves

        ll_code = self.compare_models(tree)

        # BDMS likelihood by hand.
        # The likelihood is the probability that no BDM event happens along this branch (ie. one would've happened after sample time),
        # times the probability of being sampled
        naive_ll_hand = log_dexp(event.dist, self.Λ(event.up)) + (
            log(self.μ(event.up)) - log(self.Λ(event.up))
        )

        # MTBD likelihood by hand
        present_time = max([node.t for node in tree.get_leaves()])
        appx_ll_hand = (
            self.log_f_N(event, present_time) + log(self.σ) + log(self.μ(event.up))
        )

        self.assertAlmostEqual(ll_code, naive_ll_hand, places=TOLERANCE)
        self.assertAlmostEqual(ll_code, appx_ll_hand, places=TOLERANCE)

    def test_mutation_event(self):
        """Edge with a type change, then eventually being sampled."""

        tree = bdms.TreeNode()
        tree.x = self.state_space[0]
        m_event = add_event(tree, MUTATION, edge_length=3)
        self.mutator.mutate(m_event)
        s_event = add_event(m_event, SAMPLED_SURVIVOR, edge_length=4)
        tree._sampled = True

        ll_code = self.compare_models(tree)

        # BDMS likelihood by hand.
        # The likelihood is the probability of mutating after the given time
        # (which is probability to any event, times probability the event is a mutation),
        # times the probability of the specific mutation that occurred,
        # times the likelihood as derived in "Single edge to sample time"
        naive_ll_hand = (
            log_dexp(m_event.dist, self.Λ(m_event.up))
            + (log(self.γ(m_event.up)) - log(self.Λ(m_event.up)))
            + self.mutator.logprob(m_event)
            + log_hazard_exp(s_event.dist, self.Λ(s_event.up))
            + log(self.ρ)
        )

        # MTBD likelihood by hand
        present_time = max([node.t for node in tree.get_leaves()])
        appx_ll_hand = (
            self.log_f_N(m_event, present_time)
            + self.log_f_N(s_event, present_time)
            + log(self.γ(m_event.up))
            + self.mutator.logprob(m_event)
            + log(self.ρ)
        )

        self.assertAlmostEqual(ll_code, naive_ll_hand, places=TOLERANCE)
        self.assertAlmostEqual(ll_code, appx_ll_hand, places=TOLERANCE)

    def test_birth_event(self):
        """A bifurcation with both children sampled."""

        tree = bdms.TreeNode()
        tree.x = self.state_space[0]
        b_event = add_event(tree, BIRTH, edge_length=3)

        s_events = []

        for i in range(bdms.TreeNode._OFFSPRING_NUMBER):
            # Note to self: make sure that these are the same length...
            # sampling of survivors only happens at the present
            s_events.append(add_event(b_event, SAMPLED_SURVIVOR, edge_length=4))

        tree._sampled = True

        ll_code = self.compare_models(tree)

        # BDMS likelihood by hand.
        # The likelihood is the probability of birthing after the given time
        # (which is probability to any event, times probability the event is a birth),
        # times the likelihood as derived in "Single edge to sample time", once for each child
        naive_ll_hand = (
            log_dexp(b_event.dist, self.Λ(b_event.up))
            + (log(self.λ(b_event.up)) - log(self.Λ(b_event.up)))
            + log_hazard_exp(s_events[0].dist, self.Λ(s_events[0].up))
            + log(self.ρ)
            + log_hazard_exp(s_events[1].dist, self.Λ(s_events[1].up))
            + log(self.ρ)
        )

        # MTBD likelihood by hand
        present_time = max([node.t for node in tree.get_leaves()])
        appx_ll_hand = (
            self.log_f_N(b_event, present_time)
            + self.log_f_N(s_events[0], present_time)
            + self.log_f_N(s_events[1], present_time)
            + log(self.λ(b_event.up))
            + 2 * log(self.ρ)
        )

        self.assertAlmostEqual(ll_code, naive_ll_hand, places=TOLERANCE)
        self.assertAlmostEqual(ll_code, appx_ll_hand, places=TOLERANCE)

    def test_full_tree(self):
        """A fully grown tree."""

        trees = bdms.sample_trees(n=1, t=2, init_x=1, **self.parameter_dict, seed=10)

        self.compare_models(trees[0])

    def log_f_N(self, event, present_time):
        """The logarithm of the f_N quantity in the Stadler approximate likelihood."""
        c = sqrt(
            self.Λ(event.up) ** 2
            - 4 * self.μ(event.up) * (1 - self.σ) * self.λ(event.up)
        )
        x = (-self.Λ(event.up) - c) / 2
        y = (-self.Λ(event.up) + c) / 2

        helper = (
            lambda t: (y + self.λ(event.up) * (1 - self.ρ)) * exp(-c * t)
            - x
            - self.λ(event.up) * (1 - self.ρ)
        )

        t_s = present_time - (event.t - event.dist)
        t_e = present_time - event.t

        return c * (t_e - t_s) + 2 * (log(helper(t_e)) - log(helper(t_s)))


if __name__ == "__main__":
    unittest.main()
