from gcdyn import bdms, mutators, model, responses
from numpy import log, sqrt, exp, max
from scipy.stats import expon
import unittest


BIRTH = bdms.TreeNode._BIRTH_EVENT
DEATH = bdms.TreeNode._DEATH_EVENT
MUTATION = bdms.TreeNode._MUTATION_EVENT
SAMPLED_SURVIVOR = bdms.TreeNode._SAMPLING_EVENT
UNSAMPLED_SURVIVOR = bdms.TreeNode._SURVIVAL_EVENT

# Number of decimal places to which log-likelihoods should be accurate
TOLERANCE = 2


def add_event(node, event, edge_length):
    child = bdms.TreeNode(t=node.t + edge_length, x=node.x, dist=edge_length)

    child.event = event
    node.add_child(child)

    return child


def log_dexp(x, rate):
    return expon.logpdf(x, scale=1 / rate)


def log_hazard_exp(x, rate):
    return expon.logsf(x, scale=1 / rate)


class TestMTBDLikelihood(unittest.TestCase):
    def setUp(self):
        self.λ = responses.SigmoidResponse()
        self.μ = responses.ConstantResponse(1)
        self.γ = responses.ConstantResponse(1)
        self.mutator = mutators.GaussianMutator(-1, 1)
        self.ρ = 1
        self.σ = 1
        self.Λ = lambda x: self.λ(x) + self.μ(x) + self.γ(x)

        responses.init_numpy(use_jax=True)
        responses.register_with_pytree(responses.SigmoidResponse)

    def test_sample_event(self):
        """Single edge to sample time."""

        tree = bdms.TreeNode(x=10)
        event = add_event(tree, SAMPLED_SURVIVOR, edge_length=3)
        tree._sampled = True  # We do this ourselves

        # BDMS likelihood by code
        m = model.BDMSModel(
            [tree],
            death_rate=self.μ,
            mutation_rate=self.γ,
            mutator=None,
            sampling_probability=self.ρ,
        )
        bdms_ll_code = m.log_likelihood(birth_rate=self.λ).item()

        # BDMS likelihood by hand.
        # The likelihood is the probability that no BDM event happens along this branch (ie. one would've happened after sample time),
        # times the probability of being sampled
        bdms_ll_hand = log_hazard_exp(event.dist, self.Λ(event)) + log(self.ρ)

        # MTBD likelihood by hand
        present_time = max([node.t for node in tree.get_leaves()])
        mtbd_ll_hand = self.log_f_N(event, present_time) + log(self.ρ)

        self.assertAlmostEqual(bdms_ll_code, bdms_ll_hand, places=TOLERANCE)
        self.assertAlmostEqual(bdms_ll_code, mtbd_ll_hand, places=TOLERANCE)

    def test_death_event(self):
        """Single edge dying and being sampled."""

        tree = bdms.TreeNode(x=10)
        event = add_event(tree, DEATH, edge_length=3)
        tree._sampled = True  # We do this ourselves

        # BDMS likelihood by code
        m = model.BDMSModel(
            [tree],
            death_rate=self.μ,
            mutation_rate=self.γ,
            mutator=None,
            sampling_probability=self.ρ,
        )
        bdms_ll_code = m.log_likelihood(birth_rate=self.λ).item()

        # BDMS likelihood by hand.
        # The likelihood is the probability that no BDM event happens along this branch (ie. one would've happened after sample time),
        # times the probability of being sampled
        bdms_ll_hand = log_dexp(event.dist, self.Λ(event)) + (
            log(self.μ(event)) - log(self.Λ(event))
        )

        # MTBD likelihood by hand
        present_time = max([node.t for node in tree.get_leaves()])
        mtbd_ll_hand = (
            self.log_f_N(event, present_time) + log(self.σ) + log(self.μ(event))
        )

        self.assertAlmostEqual(bdms_ll_code, bdms_ll_hand, places=TOLERANCE)
        self.assertAlmostEqual(bdms_ll_code, mtbd_ll_hand, places=TOLERANCE)

    def test_mutation_event(self):
        """Edge with a type change, then eventually being sampled."""

        tree = bdms.TreeNode(x=10)
        m_event = add_event(tree, MUTATION, edge_length=3)
        self.mutator.mutate(m_event)
        s_event = add_event(m_event, SAMPLED_SURVIVOR, edge_length=4)
        tree._sampled = True

        # BDMS likelihood by code
        m = model.BDMSModel(
            [tree],
            death_rate=self.μ,
            mutation_rate=self.γ,
            mutator=self.mutator,
            sampling_probability=self.ρ,
        )
        bdms_ll_code = m.log_likelihood(birth_rate=self.λ).item()

        # BDMS likelihood by hand.
        # The likelihood is the probability of mutating after the given time
        # (which is probability to any event, times probability the event is a mutation),
        # times the probability of the specific mutation that occurred,
        # times the likelihood as derived in "Single edge to sample time"
        bdms_ll_hand = (
            log_dexp(m_event.dist, self.Λ(m_event))
            + (log(self.γ(m_event)) - log(self.Λ(m_event)))
            + self.mutator.logprob(m_event)
            + log_hazard_exp(s_event.dist, self.Λ(s_event))
            + log(self.ρ)
        )

        # MTBD likelihood by hand
        present_time = max([node.t for node in tree.get_leaves()])
        mtbd_ll_hand = (
            self.log_f_N(m_event, present_time)
            + self.log_f_N(s_event, present_time)
            + log(self.γ(m_event))
            + self.mutator.logprob(m_event)
            + log(self.ρ)
        )

        self.assertAlmostEqual(bdms_ll_code, bdms_ll_hand, places=TOLERANCE)
        self.assertAlmostEqual(bdms_ll_code, mtbd_ll_hand, places=TOLERANCE)

    def test_birth_event(self):
        """A bifurcation with both children sampled."""

        tree = bdms.TreeNode(x=10)
        b_event = add_event(tree, BIRTH, edge_length=3)

        s_events = []

        for i in range(bdms.TreeNode._OFFSPRING_NUMBER):
            s_events.append(add_event(b_event, SAMPLED_SURVIVOR, edge_length=3 + i))

        tree._sampled = True

        # BDMS likelihood by code
        m = model.BDMSModel(
            [tree],
            death_rate=self.μ,
            mutation_rate=self.γ,
            mutator=None,
            sampling_probability=self.ρ,
        )
        bdms_ll_code = m.log_likelihood(birth_rate=self.λ).item()

        # BDMS likelihood by hand.
        # The likelihood is the probability of birthing after the given time
        # (which is probability to any event, times probability the event is a birth),
        # times the likelihood as derived in "Single edge to sample time", once for each child
        bdms_ll_hand = (
            log_dexp(b_event.dist, self.Λ(b_event))
            + (log(self.λ(b_event)) - log(self.Λ(b_event)))
            + log_hazard_exp(s_events[0].dist, self.Λ(s_events[0]))
            + log(self.ρ)
            + log_hazard_exp(s_events[1].dist, self.Λ(s_events[1]))
            + log(self.ρ)
        )

        # MTBD likelihood by hand
        present_time = max([node.t for node in tree.get_leaves()])
        mtbd_ll_hand = (
            self.log_f_N(b_event, present_time)
            + self.log_f_N(s_events[0], present_time)
            + self.log_f_N(s_events[1], present_time)
            + log(self.λ(b_event))  # + log(b_event.t)
            + 2 * log(self.ρ)
        )

        self.assertAlmostEqual(bdms_ll_code, bdms_ll_hand, places=TOLERANCE)
        self.assertAlmostEqual(bdms_ll_code, mtbd_ll_hand, places=TOLERANCE)

    def log_f_N(self, event, present_time):
        """The logarithm of the f_N quantity in the MTBD likelihood."""
        c = sqrt(self.Λ(event) ** 2 - 4 * self.μ(event) * (1 - self.σ) * self.λ(event))
        x = (-self.Λ(event) - c) / 2
        y = (-self.Λ(event) + c) / 2

        helper = (
            lambda t: (y + self.λ(event) * (1 - self.ρ)) * exp(-c * t)
            - x
            - self.λ(event) * (1 - self.ρ)
        )

        t_s = present_time - (event.t - event.dist)
        t_e = present_time - event.t

        return c * (t_e - t_s) + 2 * (log(helper(t_e)) - log(helper(t_s)))


if __name__ == "__main__":
    unittest.main()
