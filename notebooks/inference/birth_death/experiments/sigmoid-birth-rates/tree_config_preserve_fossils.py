import jax.numpy as np

from gcdyn import mutators, poisson

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
