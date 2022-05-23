"""A class to simulate evolution of a germinal center."""


import copy
from enum import Enum
import numpy as np
from gcdyn.event_sampler import EventSampler


class EventType(Enum):
    BIRTH = 0
    DEATH = 1
    MUTATION = 2


class GerminalCenter:

    r"""A class representing a germinal center simulation.

    We will use "cells" to mean ete3 nodes with specific attributes.


    Args:
        starting_population: list of cells
    """

    def __init__(self, starting_population: list):
        self.time = 0.0
        # XXX Seems like these should be _population, _rng, _sampler.
        self.population = []
        self.trees = copy.deepcopy(starting_population)
        self.rng = np.random.default_rng()
        # XXX Passing the RNG to the sampler. We want this?
        self.sampler = EventSampler(self.rng)
        for cell in self.trees:
            self.add_cell(cell)

    def print_trees(self):
        """Print the simulated trees describing the simulation."""
        for tree in self.trees:
            print(tree)

    def add_cell(self, cell):
        """Add the cell to the GC."""
        self.population.append(cell)
        self.sampler.append(np.array([cell.λ, cell.μ, cell.m]))

    def pop_cell(self, cell_idx):
        """Remove the cell from the GC and return it."""
        cell = self.population.pop(cell_idx)
        assert not cell.children, (
            "We should only be dropping cells from the current generation, not ancestral"
            " ones with descendants."
        )
        self.sampler.drop(cell_idx)
        return cell

    def make_and_add_child_cell(self, parent):
        """Make a new cell, connect it to its parent, and add it to the GC."""
        child = copy.deepcopy(parent)
        child.detach()
        # We have duplicated the parent, but we don't want to keep its descendants. This
        # is important when we are adding two descendants to a given parent.
        for fake_grandchild in child.children:
            child.remove_child(fake_grandchild)
        parent.add_child(child, dist=0.0)
        self.add_cell(child)
        return child

    def mutate_cell(self, cell):
        """Mutate a cell."""
        # TODO something not-dumb
        cell.λ = self.rng.uniform()
        cell.μ = self.rng.uniform()
        cell.m = self.rng.uniform()

    def implement_step(self, time_to_event, event_type, cell_idx):
        """Implement the next event in the simulation.

        Separated out from `step` in part because it's nice for testing
        to have something deterministic.
        """
        self.time += time_to_event
        cell = self.pop_cell(cell_idx)

        for bystander_cell in self.population:
            bystander_cell.dist += time_to_event

        if event_type == EventType.BIRTH:
            for _ in range(2):
                self.make_and_add_child_cell(cell)
                # XXX Should we mutate the cells here?
        elif event_type == EventType.DEATH:
            # XXX do we care about marking a cell as dead? Other cells that are not in
            # the population are also "dead".
            cell.dead = True
        elif event_type == EventType.MUTATION:
            mutated_cell = self.make_and_add_child_cell(cell)
            self.mutate_cell(mutated_cell)
        else:
            assert False

    def step(self):
        """Draw and implement the next event in the simulation.

        Draw the time to the next event, the event type and the the
        impacted cell and implement the modification.
        """
        time_to_event = self.sampler.sample_time_to_next_event()
        (event_idx, cell_idx) = self.sampler.sample_next_event()
        self.implement_step(time_to_event, EventType(event_idx), cell_idx)

    def run(self, stopping_time):
        """Run until we have achieved the sampling time or all the cells are
        dead."""
        while self.time < stopping_time:
            if not self.population:
                break
            self.step()

        # XXX mark sampled cells as sampled?
        # TODO cut down branch lengths to sampled time
