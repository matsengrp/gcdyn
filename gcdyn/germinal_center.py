import copy
import numpy as np
from enum import Enum


class EventType(Enum):
    BIRTH = 0
    DEATH = 1
    MUTATION = 2


class EventSampler:
    def __init__(self):
        self.rates = np.array([])

    def sample_time_to_next_event(self):
        """
        Sample the time to the next event.
        """
        pass

    def sample_next_event(self):
        """
        Sample the next event.

        Returns a tuple (cell_idx, event_type)
        """
        pass

    def drop(self, to_drop_idx: int):
        """
        Sample the next event.
        """
        pass

    def append(self, rate_column: np.ndarray):
        """
        Append some new rates.

        Return the index of the appended item.
        """
        assert len(rate_column) == len(EventType)
        self.rates = np.hstack(self.rates, rate_column.reshape(-1, 1))


class GerminalCenter:

    r"""A class representing a germinal center simulation.

    We will use "cells" to mean ete3 nodes with specific attributes.
    We could also use "events".


    Args:
        starting_population: list of cells
    """

    def __init__(self, starting_population: list):
        self.trees = copy.deepcopy(starting_population)
        self.sampler = EventSampler()
        self.time = 0.0

        for cell in population:
            self.add_cell(cell)

    def add_cell(self, cell):
        """
        Add the cell to the population and the sampler.
        """
        population.append(cell)
        sampler.append(np.array([cell.birth_rate, cell.death_rate, cell.mutation_rate]))

    def make_child_cell(cell):
        child = copy.deepcopy(cell)
        # TODO set branch length to zero
        # TODO make cell the parent of mutated cell
        # XXX Could this be "make and add child cell"?

    def step(self):
        """
        Draw the next event in the simulation.

        First draw the time to the next event.
        Then draw the impacted cell.
        Loop through the cells.
        If it's not the impacted cell, extend branch length, and if it is, implement the event.
        """

        time_to_next_event = self.sampler.sample_time_to_next_event()
        (cell_idx, event_type) = self.sampler.sample_next_event()
        self.time += time_to_next_event

        cell = population.pop(cell_idx)
        self.sampler.drop(cell_idx)
        # TODO add branch length to cell

        if event_type == BIRTH:
            child1 = self.make_child_cell(cell)
            child2 = self.make_child_cell(cell)
            self.add_cell(child1)
            self.add_cell(child2)
            # TODO connect nodes
        elif event_type == DEATH:
            # TODO do we care about marking a cell as dead? Other cells that are not in
            # the population are also "dead".
            cell.dead = True
        elif event_type == MUTATION:
            mutated_cell = copy.make_child_cell(cell)
            # TODO modify rates
            self.add_cell(mutated_cell)
        else:
            assert False

    def run(self, stopping_time):
        while self.time < stopping_time:
            self.step()

        # TODO mark sampled cells as sampled
        # TODO cut down branch lengths to sampled time
