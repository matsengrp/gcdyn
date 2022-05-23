import numpy as np
import gcdyn.event_sampler as event_sampler


def test_build_and_modify_sampler():
    rng = np.random.default_rng()
    sampler = event_sampler.EventSampler(rng)
    sampler.append(np.array([1.0, 2.0, 0.0]))
    (event, entity) = sampler.sample_next_event()
    assert entity == 0
    assert event < 2
    sampler.append(np.array([0.0, 0.0, 103.0]))
    sampler.drop(0)
    (event, entity) = sampler.sample_next_event()
    assert entity == 0
    assert event == 2
