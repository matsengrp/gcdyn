import numpy as np
import ete3
import gcdyn.germinal_center as germinal_center
from gcdyn.germinal_center import EventType


def make_random_cell():
    rng = np.random.default_rng()
    cell = ete3.Tree()
    # XXX Note that I'm not using add_feature... should I?
    cell.λ = rng.uniform()
    cell.μ = rng.uniform()
    cell.m = rng.uniform()
    return cell


def make_birthy_cell():
    rng = np.random.default_rng()
    cell = ete3.Tree()
    # XXX Note that I'm not using add_feature... should I?
    cell.λ = 2.0
    cell.μ = 0.1
    cell.m = 1.0
    return cell


def test_manual_simulation():
    gc = germinal_center.GerminalCenter([make_random_cell(), make_random_cell()])
    gc.implement_step(0.3, EventType.BIRTH, 1)
    for _ in range(8):
        gc.implement_step(0.7, EventType.MUTATION, 2)
    gc.implement_step(1.3, EventType.DEATH, 0)
    # TODO actually assert some things?
    gc.print_trees()


def test_simulation():
    # TODO should this go into a "demo notebook"?
    gc = germinal_center.GerminalCenter([make_birthy_cell(), make_birthy_cell()])
    gc.run(2)
    gc.print_trees()
