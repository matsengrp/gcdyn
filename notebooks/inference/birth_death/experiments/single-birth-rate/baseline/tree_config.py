import jax.numpy as np

from gcdyn import mutators, poisson

STATE_SPACE = (2,)
INITIAL_STATE = 2
PRESENT_TIME = 2
NUM_TREES = 15

TRUE_PARAMETERS = {
    "birth_response": poisson.ConstantResponse(1.8),
    "death_response": poisson.ConstantResponse(1),
    "mutation_response": poisson.ConstantResponse(0),
    "mutator": mutators.DiscreteMutator(
        state_space=STATE_SPACE,
        transition_matrix=np.array([[1]]),
    ),
    "extant_sampling_probability": 1,
    "extinct_sampling_probability": 0,
}
