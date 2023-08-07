import numpy as np

from gcdyn import poisson, utils

birth_rate = 2.5
death_rate = 1.1
time = 2

trees = utils.sample_trees(
    n=1000,
    t=time,
    print_info=False,
    birth_response=poisson.ConstantResponse(birth_rate),
    death_response=poisson.ConstantResponse(death_rate),
    extinct_sampling_probability=0,
    min_survivors=0,
    prune=True,
)

print(np.mean([len(tree.get_leaves()) for tree in trees]))
print(np.exp(time * (birth_rate - death_rate)))
