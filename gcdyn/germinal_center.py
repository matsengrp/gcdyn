import copy
import numpy as np
from enum import Enum
from gcdyn.event_sampler import EventSampler


class EventType(Enum):
    BIRTH = 0
    DEATH = 1
    MUTATION = 2


class GerminalCenter:

    r"""A class representing a germinal center simulation.

    We will use "cells" to mean ete3 nodes with specific attributes.
    We could also use "events".


    Args:
        starting_population: list of cells
    """

    def __init__(self, starting_population: list):
        self.time = 0.0
        self.population = []
        self.trees = copy.deepcopy(starting_population)
        self.sampler = EventSampler()

        for cell in self.trees:
            self.add_cell(cell)

    def print_trees(self):
        for tree in self.trees:
            print(tree)  # .get_ascii(attributes=["name"]))

    def add_cell(self, cell):
        """
        Add the cell to the population and the sampler.
        """
        self.population.append(cell)
        self.sampler.append(np.array([cell.λ, cell.μ, cell.m]))

    def make_and_add_child_cell(self, parent):
        child = copy.deepcopy(parent)
        parent.add_child(child, dist=0.0)
        self.add_cell(child)
        return child

    def mutate_cell(self, cell):
        # TODO something not-dumb
        cell.λ = rng.uniform()
        cell.μ = rng.uniform()
        cell.m = rng.uniform()

    def implement_step(self, time_to_event, event_type, cell_idx):
        """
        Implement the next event in the simulation.

        Loop through the cells.
        If it's not the impacted cell, extend branch length, and if it is, implement the event.
        """

        self.time += time_to_event
        cell = self.population.pop(cell_idx)
        self.sampler.drop(cell_idx)

        for bystander_cell in self.population:
            bystander_cell.dist += time_to_event

        if event_type == EventType.BIRTH.value:
            for _ in range(2):
                child = self.make_and_add_child_cell(cell)
        elif event_type == EventType.DEATH.value:
            # TODO do we care about marking a cell as dead? Other cells that are not in
            # the population are also "dead".
            cell.dead = True
        elif event_type == EventType.MUTATION.value:
            mutated_cell = cell.make_and_add_child_cell(cell)
            mutated_cell.mutate_cell()
        else:
            assert False

    def step(self):
        """
        Draw and implement the next event in the simulation.

        Draw the time to the next event, the event type and the the impacted cell and
        implement the modification.
        """

        time_to_event = self.sampler.sample_time_to_next_event()
        (event_type, cell_idx) = self.sampler.sample_next_event()

        self.implement_step(time_to_event, event_type, cell_idx)

    def run(self, stopping_time):
        while self.time < stopping_time:
            self.step()

        # TODO mark sampled cells as sampled
        # TODO cut down branch lengths to sampled time
