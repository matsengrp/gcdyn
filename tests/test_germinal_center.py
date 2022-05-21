import numpy as np
import ete3
import gcdyn.germinal_center as germinal_center


def make_random_cell():
    rng = np.random.default_rng()
    cell = ete3.Tree()
    # XXX Note that I'm not using add_feature... should I?
    cell.λ = rng.uniform()
    cell.μ = rng.uniform()
    cell.m = rng.uniform()
    return cell


def test_build_germinal_center():
    gc = germinal_center.GerminalCenter([make_random_cell(), make_random_cell()])
    gc.implement_step(0.3, 0, 1)
    gc.implement_step(0.3, 0, 1)
    gc.print_trees()
